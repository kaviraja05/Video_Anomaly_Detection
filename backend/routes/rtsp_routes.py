from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import logging
from backend.rtsp_stream import RTSPProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/rtsp-stream")
async def get_rtsp_stream(rtsp_url: str, request: Request):
    """
    Initiates an infinite MJPEG & JSON Data Server-Sent Events stream targeting the specific RTSP configuration.
    """
    logger.info(f"Starting RTSP Stream for {rtsp_url}")
    processor = RTSPProcessor(rtsp_url)
    
    async def event_generator():
        try:
            async for event in processor.stream_generator():
                if await request.is_disconnected():
                    logger.info("Client cleanly disconnected from frontend view, terminating RTSP processor.")
                    processor.is_running = False
                    break
                yield event
        except asyncio.CancelledError:
            logger.warning("RTSP Pipeline forcefully cancelled.")
            processor.is_running = False
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
