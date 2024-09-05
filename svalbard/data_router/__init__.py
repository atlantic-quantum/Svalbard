"""
Initialize the data_router sub module.
"""

from .data_router import (
    DATA_CONFIG_FILE_PATH,
    DATA_CONFIG_LOCAL_PATH,
    DATA_CONFIG_PATH,
    DATA_CONFIG_TEST_PATH,
    ds_router,
    lifespan,
)

__all__ = [
    "DATA_CONFIG_FILE_PATH",
    "DATA_CONFIG_TEST_PATH",
    "DATA_CONFIG_PATH",
    "DATA_CONFIG_LOCAL_PATH",
    "ds_router",
    "lifespan",
]
