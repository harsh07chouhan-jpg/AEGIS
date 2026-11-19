
"""AEGIS FastAPI backend for the verified audio inference service."""

from contextlib import asynccontextmanager
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

APP_NAME = "AEGIS Audio Forensics API"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Store one service instance for the lifetime of the application.
service_state = {"service": None, "startup_error": None}


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
        "Audio deepfake screening using the verified AEGIS audio model. "
        "Scores are uncalibrated model scores, not probabilities or proof "
        "of authenticity."
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
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
        await file.close()
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove temporary upload file.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
