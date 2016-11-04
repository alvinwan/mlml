import datetime
import functools


def timeit(f):
    """Times the function that it decorates."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        a = datetime.datetime.now()
        rv = f(*args, **kwargs)
        b = datetime.datetime.now()
        c = b - a
        print('Time (s):', c.total_seconds())
        return rv
    return wrapper