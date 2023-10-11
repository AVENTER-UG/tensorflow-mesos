from contextlib import contextmanager
from tfmesos2.scheduler import Job, TensorflowMesos

__VERSION__ = '0.0.10'


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
        s.wait_until_ready()
        yield s
    finally:
        s.stop()