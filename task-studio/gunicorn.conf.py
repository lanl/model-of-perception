"""One process shares visitor sessions and the bounded generation queue."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8765')}"
workers = 1
worker_class = 'gthread'
threads = 4
preload_app = False
# Generation is asynchronous; HTTP threads only submit jobs or return results.
timeout = 120
graceful_timeout = 30
accesslog = '-'
errorlog = '-'
# Session/job state lives in this worker. Do not recycle it between requests.
max_requests = 0


def worker_exit(server, worker):
    app = getattr(worker, 'wsgi', None)
    if app is not None and hasattr(app, 'extensions'):
        app.extensions['task_runtime'].close()
