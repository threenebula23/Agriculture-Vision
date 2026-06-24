from fastapi import APIRouter, status
from agro_gis.schemas import ProcessingConfig, GeoMetadata
from agro_gis.services.queue_worker import create_job, task_queue

router = APIRouter(prefix="/api/v1/maps", tags=["Maps & Layers"])


@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def process_tile(config: ProcessingConfig):
    # Создаем запись о задаче
    job = create_job(config)

    # Кладем ID задачи в асинхронную очередь для воркера
    await task_queue.put(job.id)

    return {
        "message": "Задание на обработку успешно добавлено в очередь",
        "job_id": job.id
    }

# Остальные эндпоинты (upload, objects, probabilities) остаются такими же, как в предыдущей версии.