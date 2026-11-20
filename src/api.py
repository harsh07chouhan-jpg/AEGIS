
"""AEGIS FastAPI backend for the verified audio inference service."""

from contextlib import asynccontextmanager
from threading import Lock
from collections import deque
from time import monotonic
import logging
import os
from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

# Ensure the directory containing this script is on sys.path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from audio_inference_service import AudioInferenceService, ModelIntegrityError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.api")

APP_NAME = "AEGIS Media Forensics API"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Store one service instance for the lifetime of the application.
service_state = {"service": None, "startup_error": None}

VIDEO_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
VIDEO_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_video_service = None
_video_service_lock = Lock()
# Shared concurrency guard for CPU-intensive audio/video requests.
_inference_request_lock = Lock()
# Global rate limiter for public inference requests.
RATE_LIMIT_MAX_REQUESTS = 6
RATE_LIMIT_WINDOW_SECONDS = 60

_rate_limit_lock = Lock()
_rate_limit_requests = deque()


def get_video_service():
    """Load the video model on first use, without blocking audio startup."""
    global _video_service
    if _video_service is None:
        with _video_service_lock:
            if _video_service is None:
                from video_detector.src.inference import VideoInference
                _video_service = VideoInference()
    return _video_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and verify the checkpoint once when the API starts."""
    try:
        service_state["service"] = AudioInferenceService(num_threads=2)
        service_state["startup_error"] = None
        logger.info("AEGIS audio inference service loaded and verified.")
    except Exception as exc:
        service_state["service"] = None
        service_state["startup_error"] = type(exc).__name__
        logger.exception("AEGIS model initialization failed.")
    yield
    service_state["service"] = None


app = FastAPI(
    title=APP_NAME,
    description=(
        "Audio and video deepfake screening using AEGIS research models. "
        "Model scores are not proof of authenticity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://aegis-9id.pages.dev",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
@app.middleware("http")
async def inference_rate_limit(request, call_next):
    """Limit audio and video analysis requests globally."""
    if (
        request.method == "POST"
        and request.url.path in ("/analyze/audio", "/analyze/video")
    ):
        now = monotonic()

        with _rate_limit_lock:
            while (
                _rate_limit_requests
                and now - _rate_limit_requests[0] >= RATE_LIMIT_WINDOW_SECONDS
            ):
                _rate_limit_requests.popleft()

            if len(_rate_limit_requests) >= RATE_LIMIT_MAX_REQUESTS:
                retry_after = max(
                    1,
                    int(
                        RATE_LIMIT_WINDOW_SECONDS
                        - (now - _rate_limit_requests[0])
                    ) + 1,
                )

                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": {
                            "code": "rate_limit_exceeded",
                            "message": (
                                "Too many analysis requests. "
                                "Please try again later."
                            ),
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            _rate_limit_requests.append(now)

    return await call_next(request)


def get_service() -> AudioInferenceService:
    service = service_state["service"]
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "The audio model is not ready. Check server logs.",
            },
        )
    return service


@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "status": (
            "ready" if service_state["service"] is not None else "unavailable"
        ),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ready = service_state["service"] is not None
    return {
        "status": "healthy" if ready else "unavailable",
        "model_loaded": ready,
        "startup_error": service_state["startup_error"],
    }


@app.get("/model-info")
def model_info():
    service = get_service()
    return service.model_info()


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze one uploaded audio file without retaining it on disk."""
    service = get_service()
    if not _inference_request_lock.acquire(blocking=False):
        await file.close()
        raise HTTPException(
            status_code=429,
            detail={
                "code": "inference_busy",
                "message": "Another analysis is in progress. Please try again shortly.",
            },
            headers={"Retry-After": "5"},
        )

    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix[:12]
    temp_path = None
    total_bytes = 0

    try:
        # Stream to a temporary file with a hard size limit.
        with tempfile.NamedTemporaryFile(
            prefix="aegis_upload_",
            suffix=suffix,
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "file_too_large",
                            "message": "Maximum upload size is 50 MiB.",
                        },
                    )

                temp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "empty_upload",
                    "message": "The uploaded file is empty.",
                },
            )

        # Inference is synchronous and CPU-bound; keep it off the event loop.
        result = await run_in_threadpool(
            service.analyze_file,
            temp_path,
            filename,
        )

        status = result.get("status")
        if status == "rejected":
            # Return the service's structured validation result to the client.
            return result

        if status == "error":
            logger.error(
                "Inference failed for uploaded file %r: %s",
                filename,
                result.get("error", {}).get("code", "unknown"),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "inference_failed",
                    "message": "The audio analysis could not be completed.",
                },
            )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected upload/API failure.")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "api_error",
                "message": "The request could not be processed.",
            },
        )
    finally:
        try:
            await file.close()
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Could not remove temporary upload file.")
        finally:
            _inference_request_lock.release()


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze one video upload; temporary media and visualizations are removed."""
    filename = Path(file.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_video_format",
                "message": "Supported video formats: MP4, MOV, MKV, AVI, WEBM and M4V.",
            },
        )

    if not _inference_request_lock.acquire(blocking=False):
        await file.close()
        raise HTTPException(
            status_code=429,
            detail={
                "code": "inference_busy",
                "message": "Another analysis is in progress. Please try again shortly.",
            },
            headers={"Retry-After": "5"},
        )

    total_bytes = 0
    try:
        with tempfile.TemporaryDirectory(prefix="aegis_video_") as workdir:
    
            temp_path = Path(workdir) / ("upload" + suffix)
            with temp_path.open("wb") as temp:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > VIDEO_MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "code": "file_too_large",
                                "message": "Maximum video upload size is 200 MiB.",
                            },
                        )
                    temp.write(chunk)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "empty_upload", "message": "The uploaded file is empty."},
                )

            try:
                service = await run_in_threadpool(get_video_service)
            except Exception:
                logger.exception("AEGIS video model initialization failed.")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "video_model_unavailable",
                        "message": "The video model could not be loaded. Check server logs.",
                    },
                )

            try:
                result = await run_in_threadpool(
                    service.analyze, str(temp_path), None, workdir
                )
            except Exception as exc:
                from detector_runtime import MediaTooLongError
                from video_detector.src.utils import InferenceError

                if isinstance(exc, MediaTooLongError):
                    raise HTTPException(
                        status_code=413,
                        detail={"code": "media_too_long", "message": str(exc)},
                    )
                if isinstance(exc, InferenceError):
                    raise HTTPException(
                        status_code=422,
                        detail={"code": exc.code, "message": exc.message},
                    )
                logger.exception("Unexpected video inference failure for %r.", filename)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "video_inference_failed",
                        "message": "The video analysis could not be completed.",
                    },
                )

            result.pop("local_gradcam_paths", None)
            result["media_type"] = "video"
            result["input"] = {
                "filename": filename,
                "file_size_bytes": total_bytes,
            }
            return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected video upload/API failure.")
        raise HTTPException(
            status_code=500,
            detail={"code": "video_api_error", "message": "The video request could not be processed."},
        )
    finally:
        try:
            await file.close()
        finally:
            _inference_request_lock.release()



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
