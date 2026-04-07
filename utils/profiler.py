import json
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class Profiler:
    """單例模式，儲存全域的效能數據，供各個子模組寫入延遲"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance.metrics = defaultdict(list)
        return cls._instance

    def log(self, name, duration):
        self.metrics[name].append(duration)

    def export_json(self, filepath="benchmark_result.json"):
        if not self.metrics:
            logger.info("No profiling data collected.")
            return

        summary = {}
        for name, durations in self.metrics.items():
            avg_ms = (sum(durations) / len(durations)) * 1000
            summary[name] = {
                "avg_ms": round(avg_ms, 3),
                "min_ms": round(min(durations) * 1000, 3),
                "max_ms": round(max(durations) * 1000, 3),
                "avg_fps": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0,
                "calls": len(durations),
            }

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"效能報告已匯出至 {filepath}")


# Decorator 用法
def profile_time(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            Profiler().log(name, duration)
            return result

        return wrapper

    return decorator


# Context Manager 用法
class TimeIt:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        Profiler().log(self.name, duration)
