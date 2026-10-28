import json
import logging
import re
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.database import AsyncSessionLocal
from app.audit.ledger import AuditLedger
from app.models.audit import AuditAction, AuditOutcome
from app.core.security import decode_token

logger = logging.getLogger("audit")
_CASE_UUID_REGEX = re.compile(r"/cases/([a-f0-9\-]{36})")


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware ensuring NO endpoint skips audit logging.
    Logs successful authenticated requests, application errors, and access denials.
    When a service explicitly logs a semantic action, request.state.audited is set to True
    to prevent duplicate entries.
    """
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        endpoint = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Skip logging static/docs/health checks to prevent clutter
        if endpoint in ["/docs", "/redoc", "/openapi.json", "/favicon.ico", "/health", "/api/v1/health"]:
            return await call_next(request)

        # Mark request as not yet semantically audited
        request.state.audited = False

        # Extract user & role from JWT Bearer token if present
        user_id = None
        role = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            try:
                payload = decode_token(token)
                user_id = payload.get("sub")
                role = payload.get("role")
            except Exception:
                pass

        if not user_id:
            user_id = request.headers.get("X-User-ID", None)
        if not role:
            role = request.headers.get("X-User-Role", None)

        case_id = request.headers.get("X-Case-ID", None)
        if not case_id:
            match = _CASE_UUID_REGEX.search(endpoint)
            if match:
                case_id = match.group(1)

        body_bytes = b""
        try:
            body_bytes = await request.body()
        except Exception:
            pass

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request._receive = receive

        payload_summary = {
            "query_params": dict(request.query_params),
            "method": method,
        }
        if body_bytes:
            try:
                raw_json = json.loads(body_bytes.decode("utf-8"))
                if isinstance(raw_json, dict):
                    redacted = dict(raw_json)
                    if "password" in redacted:
                        redacted["password"] = "[REDACTED]"
                    if "totp_code" in redacted:
                        redacted["totp_code"] = "[REDACTED]"
                    payload_summary["body"] = redacted
                else:
                    payload_summary["body"] = raw_json
            except Exception:
                payload_summary["body_length"] = len(body_bytes)

        status_code = 500
        action = AuditAction.API_REQUEST
        outcome = AuditOutcome.SUCCESS
        detail = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            if status_code in [401, 403]:
                action = AuditAction.ACCESS_DENIED
                outcome = AuditOutcome.DENIED
                detail = f"Access denied to {endpoint} (HTTP {status_code})"
            elif status_code >= 500:
                outcome = AuditOutcome.ERROR
                detail = f"Server error at {endpoint} (HTTP {status_code})"
            elif status_code >= 400:
                outcome = AuditOutcome.DENIED
                detail = f"Client error at {endpoint} (HTTP {status_code})"
            return response
        except Exception as exc:
            action = AuditAction.API_REQUEST
            outcome = AuditOutcome.ERROR
            detail = f"Unhandled exception: {str(exc)}"
            raise exc
        finally:
            # If the endpoint did not log a semantic audit entry, log here
            if not getattr(request.state, "audited", False):
                try:
                    async with AsyncSessionLocal() as session:
                        await AuditLedger.append_entry(
                            session=session,
                            action=action,
                            resource=endpoint,
                            outcome=outcome,
                            user_id=user_id,
                            role=role,
                            case_id=case_id,
                            ip=client_ip,
                            user_agent=user_agent,
                            details={"detail": detail, "status_code": status_code, "request": payload_summary}
                        )
                        await session.commit()
                except Exception as audit_err:
                    logger.error(f"Audit log recording error: {audit_err}", exc_info=True)
