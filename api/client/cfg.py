"""Configuration options for API Client."""
import logging

MAX_RETRIES = 4  # Try requests 5 times before raising an exception
DEFAULT_LOG_LEVEL = logging.ERROR  # change to DEBUG for more detail
MAX_CONCURRENT_REQUESTS = 240  # batch_client will spawn this many threads
TIMEOUT = 6000  # wait this long for server response before raising a timeout exception
# Take the cartesian product of the first N search results of each type
MAX_RESULT_COMBINATION_DEPTH = 3
HOST = 'https://api.gro-intelligence.com'  # default host unless otherwise specified
