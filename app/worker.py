from redis import Redis
from rq import Worker, Queue

listen = ['flux_image_gen']
redis_conn = Redis(host='redis', port=6379)

if __name__ == '__main__':
    queues = [Queue(name, connection=redis_conn) for name in listen]
    worker = Worker(queues, connection=redis_conn, default_job_timeout=1200)
    worker.work()
