from redis import Redis
from rq import Worker, Queue, Connection

listen = ['flux_image_gen']
redis_conn = Redis(host='redis', port=6379)

if __name__ == '__main__':
    with Connection(redis_conn):
        queues = [Queue(name, connection=redis_conn) for name in listen]
        worker = Worker(queues)
        worker.work()
