import logging
from functools import wraps
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamp, level, and message
    level=logging.INFO,  # Default level
)

logger = logging.getLogger(__name__)  # Create a logger for the module

def debug_mode(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for a 'debug' keyword argument
        debug = kwargs.get("debug", False)
        if debug:
            print("debug mode:")
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Call the original function
        return func(*args, **kwargs)

    return wrapper