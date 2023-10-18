from contextlib import contextmanager
from tfmesos2.scheduler import Job, TensorflowMesos, API

@contextmanager
def cluster(jobs, **kw):
    if isinstance(jobs, dict):
        jobs = [Job(**jobs)]

    if isinstance(jobs, Job):
        jobs = [jobs]

    jobs = [job if isinstance(job, Job) else Job(**job)
            for job in jobs]
    try:
        s = TensorflowMesos(jobs, **kw)
        api = API(s.tasks, port=11000)
        api.start()
        s.start()
        s.wait_until_ready()
        yield s
    finally:
        s.shutdown()
