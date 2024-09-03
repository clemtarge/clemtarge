
from functools import wraps
import time

class Profiling:
    """
    >>> profiler = Profiling() # instantiate profiler
    ## add @profiler as decorator at each function you want to profile
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

        # Table headers
        l1 = max(map(len, self.total_time.keys()))
        l2 = 12
        message += f"{'Function':^{l1}} {'Total time':^{l2}} {'Num calls':^{l2}} {'Mean time':^{l2}} {'% of global time':^{l2}} \n\n"
        
        keys = list(self.total_time.keys())
        num_rows = len(keys)
        num_cols = 5
        table = [[None] * num_rows for _ in range(num_cols)]
        
        # Fill the table
        for i, key in enumerate(keys):
            table[0][i] = f"{key:^{l1}}"
            table[1][i] = f"{f'{self.total_time[key]:.3f} s':^{l2}}"
            table[2][i] = f"{self.n_calls[key]:^{l2}}"
            table[3][i] = f"{f'{self.total_time[key] / self.n_calls[key]:.3f} s':^{l2}}"
            table[4][i] = f"{f'{100*self.total_time[key] / total_time:.2f} %':^{l2}}"
        
        # Print the transposed table
        for row in list(map(list, zip(*table))):
            message += " ".join(row) + "\n"
        
        return message