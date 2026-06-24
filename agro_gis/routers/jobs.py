from fastapi import APIRouter, HTTPException
from agro_gis.schemas import JobItem
from typing import List
from agro_gis.services.queue_worker import jobs_db

router = APIRouter(prefix="/api/v1/jobs", tags=["Job Queue"])

@router.get("/", response_model=List[JobItem])
async def get_jobs_queue():
    # Возвращаем все задачи, отсортированные по времени создания (новые сверху)
    sorted_jobs = sorted(jobs_db.values(), key=lambda x: x.created_at, reverse=True)
    return sorted_jobs

@router.get("/{job_id}", response_model=JobItem)
async def get_job_status(job_id: str):
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job