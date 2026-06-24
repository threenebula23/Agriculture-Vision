import asyncio
import uuid
import random
from datetime import datetime
from agro_gis.schemas import JobItem, ProcessingConfig

# In-memory хранилище задач: {job_id: JobItem}
jobs_db = {}
# Асинхронная очередь для передачи ID задач воркеру
task_queue = asyncio.Queue()


def create_job(config: ProcessingConfig) -> JobItem:
    job_id = str(uuid.uuid4())

    # Определяем тип задачи на основе конфига
    tasks = []
    if config.detect_boundaries: tasks.append("Сегментация")
    if config.detect_points: tasks.append("Детекция точек")
    if config.classify_crops: tasks.append("Классификация")
    task_type = " + ".join(tasks) if tasks else "Неизвестная задача"

    new_job = JobItem(
        id=job_id,
        filename=config.filename,
        task_type=task_type,
        status="Ожидание",
        created_at=datetime.now()
    )
    jobs_db[job_id] = new_job
    return new_job


async def process_job(job_id: str):
    job = jobs_db.get(job_id)
    if not job:
        return

    # Переводим в работу
    job.status = "В процессе"
    job.started_at = datetime.now()

    # Мокаем работу нейронки (задержка от 5 до 15 секунд)
    processing_time = random.randint(5, 15)
    await asyncio.sleep(processing_time)

    # Завершаем задачу
    job.status = "Готово"
    job.completed_at = datetime.now()
    job.duration_seconds = int((job.completed_at - job.started_at).total_seconds())


async def worker():
    """Фоновый воркер, который постоянно читает очередь и выполняет задачи."""
    while True:
        job_id = await task_queue.get()
        try:
            await process_job(job_id)
        except Exception as e:
            # Обработка ошибок, если "нейронка" упадет
            if job_id in jobs_db:
                jobs_db[job_id].status = "Ошибка"
        finally:
            task_queue.task_done()