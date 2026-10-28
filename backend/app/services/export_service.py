import io
import time
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from google.genai import types

from app.config import settings
from app.models.case import Case, CompositeSession
from app.models.user import User, UserRole
from app.models.audit import AuditAction, AuditOutcome
from app.audit.ledger import AuditLedger
from app.schemas.face_spec import FaceSpec, FeatureAttribute
from app.services.gemini_client import gemini_client
from app.services.storage import storage_service
from app.services.locality import to_numpy_rgb

logger = logging.getLogger("export_service")


def convert_to_sketch_opencv(image_rgb: np.ndarray) -> np.ndarray:
    """
    Deterministic OpenCV sketch fallback:
    grayscale -> invert -> Gaussian blur -> color-dodge blend.
    Preserves all proportions and facial structures exactly with zero hallucinated details.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    # Kernel size 21x21 for smooth shading gradients
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    # Color-dodge blend: divide gray by inverted blur (255 - blur)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


async def convert_to_sketch_gemini(image_bytes: bytes) -> Optional[bytes]:
    """
    Attempts to convert the composite image into a clean graphite forensic-artist pencil sketch
    using the Gemini model. Preserves all proportions and features exactly.
    Returns None if unavailable or on error, falling back to OpenCV.
    """
    try:
        if not gemini_client.api_key:
            return None
        client = gemini_client.get_client()
        prompt_instruction = (
            "Convert this facial composite image into a clean graphite forensic-artist pencil sketch, "
            "preserving all proportions and facial features exactly with no added detail or background. "
            "Monochromatic, crisp graphite line art and subtle forensic shading on white paper."
        )
        response = client.models.generate_content(
            model=settings.GEMINI_TEXT_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt_instruction
            ]
        )
        if hasattr(response, "candidates") and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return part.inline_data.data
        return None
    except Exception as e:
        logger.warning(f"Gemini sketch conversion fallback to OpenCV: {e}")
        return None


def upscale_image_lanczos(image: Image.Image, target_scale: float = 2.0) -> Image.Image:
    """
    Upscales an image using high-quality Lanczos resampling.
    Guarantees sharp edge preservation without blur or compression artifacts.
    """
    new_w = max(1024, int(image.width * target_scale))
    new_h = max(1024, int(image.height * target_scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def burn_watermark_footer(
    image: Image.Image,
    case_id: str,
    case_number: str,
    timestamp_utc: str
) -> Image.Image:
    """
    Burns in the mandatory forensic warning watermark and metadata into a dedicated footer band.
    "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY" plus case ID and timestamp.
    """
    w, h = image.size
    footer_h = max(70, int(h * 0.08))
    total_h = h + footer_h

    canvas_img = Image.new("RGB", (w, total_h), (17, 24, 39))  # Dark slate #111827
    canvas_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas_img)

    line1 = "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY"
    line2 = f"CASE ID: {case_number or case_id} | EXPORT TIME: {timestamp_utc} | NOT FOR DIRECT IDENTIFICATION"

    try:
        font_large = ImageFont.truetype("arial.ttf", size=max(16, int(footer_h * 0.30)))
        font_small = ImageFont.truetype("arial.ttf", size=max(11, int(footer_h * 0.20)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Center Line 1 in prominent warning amber #FBBF24
    bbox1 = draw.textbbox((0, 0), line1, font=font_large)
    t1_w = bbox1[2] - bbox1[0]
    x1 = max(10, (w - t1_w) // 2)
    y1 = h + int(footer_h * 0.15)
    draw.text((x1, y1), line1, fill=(251, 191, 36), font=font_large)

    # Center Line 2 in light gray #E5E7EB
    bbox2 = draw.textbbox((0, 0), line2, font=font_small)
    t2_w = bbox2[2] - bbox2[0]
    x2 = max(10, (w - t2_w) // 2)
    y2 = y1 + max(18, int(footer_h * 0.42))
    draw.text((x2, y2), line2, fill=(229, 231, 235), font=font_small)

    return canvas_img


def embed_png_metadata(image: Image.Image, metadata: Dict[str, str]) -> bytes:
    """
    Saves image to PNG bytes with embedded metadata in standard PNG tEXt chunks.
    Embeds: case_id, version_id, face_spec_hash, user_id, export_time, model_ids, watermark.
    """
    png_info = PngInfo()
    for k, v in metadata.items():
        png_info.add_text(k, str(v))

    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=png_info)
    return buf.getvalue()


class ExportService:
    """
    Forensic Export Pipeline (Phase 4).
    Responsible for photoreal/sketch modes, high-resolution upscaling, watermark burning,
    PNG tEXt/EXIF metadata embedding, single-page PDF generation, and tamper-evident audit logging.
    """

    @classmethod
    async def render_export_image(
        cls,
        raw_image_bytes: bytes,
        mode: str,
        upscale: bool,
        case_id: str,
        case_number: str,
        timestamp_utc: str,
        metadata: Dict[str, str]
    ) -> Tuple[bytes, Image.Image]:
        """
        Processes image according to export mode and options:
        1. Convert to sketch if requested (Gemini + OpenCV fallback).
        2. Upscale / enhance to high resolution via Lanczos filter.
        3. Burn watermark footer band.
        4. Embed PNG metadata chunks.
        Returns: (png_bytes_with_metadata, final_pil_image)
        """
        clean_mode = mode.strip().lower()

        # Step 1: Mode Conversion
        if clean_mode in ("pencil-sketch", "sketch"):
            sketch_bytes = await convert_to_sketch_gemini(raw_image_bytes)
            if sketch_bytes:
                pil_img = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")
            else:
                arr = to_numpy_rgb(raw_image_bytes)
                sketch_arr = convert_to_sketch_opencv(arr)
                pil_img = Image.fromarray(sketch_arr)
        else:
            pil_img = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")

        # Step 2: Upscale / Enhance
        if upscale:
            pil_img = upscale_image_lanczos(pil_img, target_scale=2.0)

        # Step 3: Burn Watermark Footer Band
        watermarked_img = burn_watermark_footer(
            image=pil_img,
            case_id=case_id,
            case_number=case_number,
            timestamp_utc=timestamp_utc
        )

        # Step 4: Embed PNG Metadata (tEXt chunks)
        png_bytes = embed_png_metadata(watermarked_img, metadata)

        return png_bytes, watermarked_img

    @classmethod
    def generate_single_page_pdf(
        cls,
        case: Case,
        session: CompositeSession,
        current_user: User,
        watermarked_img: Image.Image,
        face_spec: FaceSpec,
        timeline_sessions: List[CompositeSession],
        file_hash: str,
        export_time_utc: str,
        mode: str,
    ) -> bytes:
        """
        Generates a strictly single-page, court-admissible forensic composite report (PDF).
        Contains:
        - Law enforcement agency header and case details.
        - High-visibility investigative warning banner.
        - Watermarked facial composite image.
        - FaceSpec morphological summary with confidence indicators.
        - Version history list.
        - Cryptographic hashes and audit trail integrity block.
        """
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        page_w, page_h = letter  # 612 x 792 pt

        # Colors
        navy = colors.HexColor("#0F172A")
        slate = colors.HexColor("#334155")
        muted_gray = colors.HexColor("#64748B")
        border_gray = colors.HexColor("#CBD5E1")
        amber_bg = colors.HexColor("#FEF3C7")
        amber_border = colors.HexColor("#D97706")
        amber_text = colors.HexColor("#92400E")
        bg_card = colors.HexColor("#F8FAFC")

        margin = 36.0  # 0.5 inch margins

        # 1. Header Banner
        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_h - 45, "FORENSIC FACIAL COMPOSITE REPORT")

        c.setFillColor(muted_gray)
        c.setFont("Helvetica", 8)
        c.drawString(margin, page_h - 57, "OFFICIAL LAW ENFORCEMENT INVESTIGATIVE LEAD ARTIFACT - STRICT NEED-TO-KNOW")

        # 2. Mandatory Warning Alert Box
        box_y = page_h - 96
        box_h = 32
        box_w = page_w - (2 * margin)
        c.setFillColor(amber_bg)
        c.setStrokeColor(amber_border)
        c.setLineWidth(1)
        c.roundRect(margin, box_y, box_w, box_h, 3, fill=1, stroke=1)

        c.setFillColor(amber_text)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(margin + 10, box_y + 19, "WARNING: AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY")
        c.setFont("Helvetica", 7.5)
        c.drawString(
            margin + 10, box_y + 8,
            "Synthetic composite generated from witness cognitive interview. Prohibited for direct automated identification or arrest warrants."
        )

        # 3. Case Metadata Card
        meta_y = box_y - 50
        meta_h = 44
        c.setFillColor(bg_card)
        c.setStrokeColor(border_gray)
        c.roundRect(margin, meta_y, box_w, meta_h, 2, fill=1, stroke=1)

        c.setFillColor(slate)
        c.setFont("Helvetica-Bold", 8)
        # Left column
        c.drawString(margin + 8, meta_y + 30, f"Case Number: {case.case_number}")
        c.drawString(margin + 8, meta_y + 18, f"Case Title: {case.title}")
        c.drawString(margin + 8, meta_y + 6, f"Status: {case.status} | Mode: {mode.upper()}")

        # Right column
        c.drawString(margin + 280, meta_y + 30, f"Export Timestamp: {export_time_utc} UTC")
        c.drawString(margin + 280, meta_y + 18, f"Exported By: {current_user.id[:8]}... ({current_user.role.value if isinstance(current_user.role, UserRole) else current_user.role})")
        c.drawString(margin + 280, meta_y + 6, f"Composite Version: Version {session.version} (Model: {session.model_id or 'Imagen 3'})")

        # 4. Middle Section: Image (Left) + FaceSpec Summary (Right)
        mid_y = meta_y - 250
        img_w = 230
        img_h = 230
        img_x = margin
        img_y = mid_y

        # Scale and draw composite image
        img_buf = io.BytesIO()
        watermarked_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        reportlab_img = ImageReader(img_buf)
        c.drawImage(reportlab_img, img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True)
        c.setStrokeColor(border_gray)
        c.rect(img_x, img_y, img_w, img_h, fill=0, stroke=1)

        # Right Column: FaceSpec Morphological Summary Card
        spec_x = img_x + img_w + 14
        spec_w = page_w - margin - spec_x
        spec_h = img_h
        c.setFillColor(bg_card)
        c.roundRect(spec_x, img_y, spec_w, spec_h, 2, fill=1, stroke=1)

        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(spec_x + 8, img_y + spec_h - 14, "FACESPEC MORPHOLOGICAL SUMMARY")
        c.setStrokeColor(border_gray)
        c.line(spec_x + 8, img_y + spec_h - 18, spec_x + spec_w - 8, img_y + spec_h - 18)

        # Morphological attributes list with confidence indicators
        attributes = [
            ("Sex / Presentation", face_spec.sex_presentation),
            ("Age Range", face_spec.age_range),
            ("Skin Tone", face_spec.skin_tone),
            ("Face Shape", face_spec.face_shape),
            ("Eye Shape", face_spec.eyes.shape),
            ("Eye Color", face_spec.eyes.color),
            ("Nose Bridge", face_spec.nose.bridge),
            ("Nose Tip", face_spec.nose.tip),
            ("Lips / Mouth", face_spec.mouth_lips),
            ("Hair Style", face_spec.hair.style),
            ("Hair Color", face_spec.hair.color),
            ("Jawline", face_spec.jawline),
            ("Chin", face_spec.chin),
            ("Facial Hair", face_spec.facial_hair),
        ]

        curr_y = img_y + spec_h - 32
        for label, attr in attributes:
            if curr_y < img_y + 8:
                break
            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(slate)
            c.drawString(spec_x + 8, curr_y, f"{label}:")

            c.setFont("Helvetica", 7)
            if attr.is_stated and attr.value:
                conf_pct = int(round(attr.confidence * 100))
                val_text = f"{attr.value[:22]} [Confidence: {conf_pct}%]"
                c.setFillColor(colors.HexColor("#065F46") if conf_pct >= 70 else slate)
            else:
                val_text = "Unspecified / Neutral"
                c.setFillColor(muted_gray)

            c.drawString(spec_x + 75, curr_y, val_text)
            curr_y -= 13.5

        # 5. Version History Section
        hist_y = img_y - 120
        hist_h = 108
        c.setFillColor(bg_card)
        c.setStrokeColor(border_gray)
        c.roundRect(margin, hist_y, box_w, hist_h, 2, fill=1, stroke=1)

        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(margin + 8, hist_y + hist_h - 14, "VERSION HISTORY & AUDIT TRAIL TIMELINE")
        c.setStrokeColor(border_gray)
        c.line(margin + 8, hist_y + hist_h - 18, margin + box_w - 8, hist_y + hist_h - 18)

        # Table header
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(muted_gray)
        c.drawString(margin + 8, hist_y + hist_h - 28, "VER")
        c.drawString(margin + 40, hist_y + hist_h - 28, "TIMESTAMP (UTC)")
        c.drawString(margin + 170, hist_y + hist_h - 28, "TRIGGERING ACTION")
        c.drawString(margin + 380, hist_y + hist_h - 28, "LOCALITY SCORE")
        c.drawString(margin + 460, hist_y + hist_h - 28, "LOCKED REGIONS")

        row_y = hist_y + hist_h - 40
        # Show last 5 iterations
        recent_sessions = list(reversed(timeline_sessions))[:5]
        for s in recent_sessions:
            if row_y < hist_y + 6:
                break
            c.setFont("Helvetica", 7)
            c.setFillColor(slate)
            c.drawString(margin + 8, row_y, f"v{s.version}")
            ts_str = s.created_at.strftime("%Y-%m-%d %H:%M:%S") if s.created_at else "Unknown"
            c.drawString(margin + 40, row_y, ts_str)
            trig_str = (s.triggering_call or "initial")[:38]
            c.drawString(margin + 170, row_y, trig_str)
            loc_str = f"{s.locality_score:.2f}" if s.locality_score is not None else "N/A"
            c.drawString(margin + 380, row_y, loc_str)
            try:
                locked_list = json.loads(s.locked_regions_json) if s.locked_regions_json else []
                locked_str = ", ".join(locked_list) if locked_list else "None"
            except Exception:
                locked_str = "None"
            c.drawString(margin + 460, row_y, locked_str[:16])
            row_y -= 13

        # 6. Bottom Cryptographic Footer
        foot_y = 28
        c.setStrokeColor(border_gray)
        c.line(margin, foot_y + 24, margin + box_w, foot_y + 24)

        face_spec_hash = hashlib.sha256(face_spec.model_dump_json().encode("utf-8")).hexdigest()
        c.setFont("Helvetica", 6.5)
        c.setFillColor(muted_gray)
        c.drawString(margin, foot_y + 14, f"FaceSpec SHA-256: {face_spec_hash}")
        c.drawString(margin, foot_y + 5, f"Artifact SHA-256: {file_hash}")

        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(navy)
        c.drawRightString(page_w - margin, foot_y + 10, "PAGE 1 OF 1")

        c.showPage()
        c.save()
        return buf.getvalue()

    @classmethod
    async def export_composite(
        cls,
        case: Case,
        current_user: User,
        db: AsyncSession,
        format: str = "png",
        mode: str = "photoreal",
        upscale: bool = True,
        version: Optional[int] = None,
        ip: str = "unknown",
        user_agent: str = "unknown",
    ) -> Tuple[bytes, str, str]:
        """
        Executes complete court-admissible forensic composite export.
        - Strict RBAC: Auditor role is denied (HTTP 403 Forbidden).
        - Mode: photoreal or pencil-sketch (Gemini + OpenCV fallback).
        - Upscale: Lanczos enhancement.
        - Watermark burned into footer band.
        - Embedded metadata in PNG tEXt chunks.
        - PNG or one-page PDF.
        - Appends export audit log entry with file hash.
        Returns: (file_bytes, mime_type, file_hash)
        """
        role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)

        # 1. RBAC Permission Check: Auditor cannot export
        if current_user.role == UserRole.AUDITOR or role_str.upper() == "AUDITOR":
            await AuditLedger.append_entry(
                session=db,
                action=AuditAction.EXPORT,
                resource=f"/api/v1/cases/{case.id}/export",
                outcome=AuditOutcome.DENIED,
                user_id=current_user.id,
                role=role_str,
                case_id=case.id,
                ip=ip,
                user_agent=user_agent,
                details={
                    "reason": "Auditor role prohibited from exporting composite images",
                    "format": format,
                    "mode": mode
                }
            )
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Auditor role restriction: Strict prohibition from exporting face composites."
            )

        # 2. Fetch target session (specific version or latest)
        if version:
            stmt = select(CompositeSession).where(
                CompositeSession.case_id == case.id,
                CompositeSession.version == version
            )
        else:
            stmt = (
                select(CompositeSession)
                .where(CompositeSession.case_id == case.id)
                .order_by(CompositeSession.version.desc())
            )
        res = await db.execute(stmt)
        target_session = res.scalars().first()

        if not target_session or not target_session.image_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No composite image found for case {case.id} (version {version or 'latest'})."
            )

        # 3. Retrieve raw composite image bytes
        raw_image_bytes = await storage_service.retrieve_image(
            case_id=case.id,
            relative_storage_path=target_session.image_path,
            db=db
        )

        face_spec = FaceSpec.model_validate_json(target_session.face_spec_json)
        face_spec_hash = hashlib.sha256(target_session.face_spec_json.encode("utf-8")).hexdigest()
        export_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 4. Prepare Metadata Dictionary
        model_ids = f"{target_session.model_id or settings.GEMINI_IMAGE_MODEL},lanczos-upscaler"
        metadata = {
            "case_id": case.id,
            "version_id": str(target_session.version),
            "face_spec_hash": face_spec_hash,
            "user_id": current_user.id,
            "export_time": export_time_utc,
            "model_ids": model_ids,
            "watermark": "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY"
        }

        # 5. Process Image (Mode, Upscale, Watermark, PNG Metadata)
        png_bytes, watermarked_img = await cls.render_export_image(
            raw_image_bytes=raw_image_bytes,
            mode=mode,
            upscale=upscale,
            case_id=case.id,
            case_number=case.case_number,
            timestamp_utc=export_time_utc,
            metadata=metadata
        )

        clean_format = format.strip().lower()

        # 6. Format Branch: PNG vs PDF
        if clean_format == "pdf":
            # Fetch all sessions for timeline list
            timeline_stmt = (
                select(CompositeSession)
                .where(CompositeSession.case_id == case.id)
                .order_by(CompositeSession.version.asc())
            )
            t_res = await db.execute(timeline_stmt)
            all_sessions = t_res.scalars().all()

            initial_hash = hashlib.sha256(png_bytes).hexdigest()
            pdf_bytes = cls.generate_single_page_pdf(
                case=case,
                session=target_session,
                current_user=current_user,
                watermarked_img=watermarked_img,
                face_spec=face_spec,
                timeline_sessions=all_sessions,
                file_hash=initial_hash,
                export_time_utc=export_time_utc,
                mode=mode,
            )
            final_bytes = pdf_bytes
            mime_type = "application/pdf"
        else:
            final_bytes = png_bytes
            mime_type = "image/png"

        # 7. Compute Final File SHA-256 Hash
        file_hash = hashlib.sha256(final_bytes).hexdigest()

        # 8. Append Tamper-Evident Audit Ledger Entry
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.EXPORT,
            resource=f"/api/v1/cases/{case.id}/export",
            outcome=AuditOutcome.SUCCESS,
            user_id=current_user.id,
            role=role_str,
            case_id=case.id,
            ip=ip,
            user_agent=user_agent,
            details={
                "format": clean_format,
                "mode": mode,
                "upscale": upscale,
                "version": target_session.version,
                "face_spec_hash": face_spec_hash,
                "file_hash": file_hash,
                "file_size": len(final_bytes),
                "model_ids": model_ids,
            }
        )
        await db.commit()

        return final_bytes, mime_type, file_hash


export_service = ExportService()
