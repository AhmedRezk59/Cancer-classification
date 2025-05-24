import functools
import logging
from time import perf_counter
from typing import Callable,Any

logger:logging.Logger = logging.getLogger(__name__)

def with_logging(func:Callable) -> Callable:
    """
    Decorator to add logging to a function.
    Args:
        func (Callable): The function to decorate.
    Returns:
        Callable: The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args:Any,**kwargs:Any)->Any:
        logger.info("Calling %s" , func.__name__)
        value = func(*args,**kwargs)
        logger.info("Finished %s" , func.__name__)
        return value
    return wrapper

def benchmark(func:Callable) ->Callable:
    """
    Decorator to benchmark a function.
    Args:
        func (Callable): The function to decorate.
    Returns:
        Callable: The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args:Any,**kwargs:Any)->Any:
        start_time = perf_counter()
        value = func(*args,**kwargs)
        end_time = perf_counter()
        logger.info("Execution time of %s: %s seconds" , func.__name__ , end_time - start_time)
        return value
    return wrapper