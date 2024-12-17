# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Only keep essential metrics
QUERY_COUNTER = Counter(
    name='rag_queries_total',
    documentation='Total number of queries',
    labelnames=['status']
)

QUERY_TIME = Histogram(
    name='rag_query_duration_seconds',
    documentation='Time spent processing queries',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]  # Fewer buckets
)

ERROR_COUNTER = Counter(
    name='rag_errors_total',
    documentation='Total number of errors',
    labelnames=['type']
)

def track_query_metrics(func):
    """Lightweight decorator for tracking essential metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            QUERY_COUNTER.labels(status='success').inc()
            return result
        except Exception as e:
            QUERY_COUNTER.labels(status='error').inc()
            ERROR_COUNTER.labels(type=type(e).__name__).inc()
            raise
        finally:
            QUERY_TIME.observe(time.time() - start_time)
    return wrapper