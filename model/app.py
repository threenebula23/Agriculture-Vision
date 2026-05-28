"""HTTP API для сегментации полей"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from model.jobs import JobQueue
from model.pipeline import run_segmentation
from model.runtime import SegmentationRuntime
from model.schemas import HealthResponse, JobStatusResponse, SegmentRequest, SegmentResponse
from model.settings import load_settings

_runtime: SegmentationRuntime | None = None
_jobs: JobQueue | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime, _jobs
    settings = load_settings()
    _runtime = SegmentationRuntime(settings)
    try:
        _runtime.load()
    except FileNotFoundError as exc:
        print(f"[model] warning: {exc}")
    _jobs = JobQueue(max_workers=settings.max_concurrent_inferences)
    yield
    if _runtime is not None:
        _runtime.unload()


app = FastAPI(
    title="Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_runtime() -> SegmentationRuntime:
    if _runtime is None:
        raise HTTPException(503, "Runtime not initialized")
    return _runtime


def _check_upload_size(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise HTTPException(
            413,
            f"Upload too large ({len(data)} bytes, max {max_bytes})",
        )


def _run_inference(
    rt: SegmentationRuntime,
    rgb_bytes: bytes,
    nir_bytes: bytes | None,
    req: SegmentRequest,
) -> SegmentResponse:
    sem = rt.acquire_inference_slot()
    with sem:
        bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid RGB image")
        rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        nir_arr = None
        if nir_bytes:
            nir_arr = cv2.imdecode(
                np.frombuffer(nir_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if nir_arr is None:
                raise ValueError("Invalid NIR image")
        return run_segmentation(rt, rgb=rgb_arr, nir=nir_arr, request=req)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rt = _get_runtime()
    return HealthResponse(
        status="ok",
        model_loaded=rt.is_loaded,
        device=str(rt.device) if rt.is_loaded else None,
        fp16=rt.meta.get("fp16_active") if rt.is_loaded else None,
        checkpoint=rt.meta.get("checkpoint_path") if rt.is_loaded else None,
    )


@app.get("/ready")
def ready() -> JSONResponse:
    rt = _get_runtime()
    if not rt.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "checkpoint not loaded"},
        )
    return JSONResponse({"ready": True})


@app.post("/v1/segment")
async def segment(
    rgb: Annotated[UploadFile, File(description="RGB image (JPEG/PNG)")],
    nir: Annotated[UploadFile | None, File(description="Optional NIR grayscale")] = None,
    threshold: float | None = Form(None),
    tta: bool | None = Form(None),
    use_sliding: bool = Form(False),
    include_mask_png: bool = Form(False),
    include_geojson: bool | None = Form(None),
    wait: bool = Form(
        True,
        description=(
            "true — ответ сразу с результатом (удобно для curl/малых снимков); "
            "false — job_id и опрос GET /v1/jobs/{id} (для долгого инференса)"
        ),
    ),
) -> SegmentResponse | dict[str, str]:
    """
    Один эндпоинт сегментации.

    Параметр `wait` вместо двух разных URL: долгий прогон не обязан держать HTTP-соединение.
    """
    rt = _get_runtime()
    if not rt.is_loaded:
        raise HTTPException(503, "Model weights not loaded")
    if _jobs is None:
        raise HTTPException(503, "Job queue not initialized")

    rgb_bytes = await rgb.read()
    _check_upload_size(rgb_bytes, rt.settings.max_upload_bytes)

    nir_bytes: bytes | None = None
    if nir is not None:
        nir_bytes = await nir.read()
        _check_upload_size(nir_bytes, rt.settings.max_upload_bytes)

    req = SegmentRequest(
        threshold=threshold,
        tta=tta,
        use_sliding=use_sliding,
        include_mask_png=include_mask_png,
        include_geojson=include_geojson,
    )

    if wait:
        try:
            return _run_inference(rt, rgb_bytes, nir_bytes, req)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"Inference failed: {exc}") from exc

    job_id = _jobs.submit(lambda: _run_inference(rt, rgb_bytes, nir_bytes, req))
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    if _jobs is None:
        raise HTTPException(503, "Job queue not initialized")
    status = _jobs.get(job_id)
    if status is None:
        raise HTTPException(404, "Job not found")
    return status
