"""Setting up logging for the module"""

import logging

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)
# using uvicorn logger here such that it logs to console when running the data_server
# as uvicorn app
