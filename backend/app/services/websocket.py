import json
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket

logger = logging.getLogger("generation_websocket")


class GenerationWebSocketManager:
    """
    Manages active WebSocket connections for streaming real-time forensic generation progress.
    Tracks connections per case_id.
    """

    def __init__(self):
        # case_id -> Set[WebSocket]
        self._active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, case_id: str, websocket: WebSocket):
        await websocket.accept()
        if case_id not in self._active_connections:
            self._active_connections[case_id] = set()
        self._active_connections[case_id].add(websocket)
        logger.info(f"WebSocket client connected for case {case_id}. Total: {len(self._active_connections[case_id])}")

    def disconnect(self, case_id: str, websocket: WebSocket):
        if case_id in self._active_connections:
            self._active_connections[case_id].discard(websocket)
            if not self._active_connections[case_id]:
                del self._active_connections[case_id]
        logger.info(f"WebSocket client disconnected for case {case_id}")

    async def broadcast(self, case_id: str, data: Dict[str, Any]):
        """
        Broadcasts generation event (queued -> generating -> done/failed) to all connected clients for the case.
        """
        if case_id not in self._active_connections:
            return

        dead_connections = set()
        message_json = json.dumps(data)

        for ws in self._active_connections[case_id]:
            try:
                await ws.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message to client in case {case_id}: {str(e)}")
                dead_connections.add(ws)

        for dead in dead_connections:
            self.disconnect(case_id, dead)


ws_manager = GenerationWebSocketManager()
