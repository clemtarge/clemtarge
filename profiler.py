
from functools import wraps
import time

class Profiling:
    """
    >>> profiler = Profiling() # instantiate profiler
    add @profiler as decorator at each function you want to profile
    >>> print(profiler) # display results
    """
    def __init__(self):
        self.total_time = {}
        self.n_calls = {}

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time

            if func.__name__ not in self.total_time:
                self.total_time[func.__name__] = 0.
                self.n_calls[func.__name__] = 0
            self.total_time[func.__name__] += total_time
            self.n_calls[func.__name__] += 1

            return result
        return wrapper

    def __str__(self) -> str:
        total_time = sum(self.total_time.values())
        message = ""
        for key in self.total_time.keys():
            message += f"In {key}:\n"
            message += f"  Total time: {self.total_time[key]:.3f}s\n"
            message += f"  Num calls: {self.n_calls[key]}\n"
            message += f"  Mean time: {self.total_time[key] / self.n_calls[key]:.3f}s\n"
            message += f"  {100*self.total_time[key] / total_time:.2f}% of global time\n\n"
        return message
