"""
Initialize the svalbard package.
"""

__all__ = ["data_model", "data_router", "data_server", "utility", "launch"]

import os

from . import data_model, data_router, data_server, utility
from .launch import launch

if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(
        "~/.aq_config/aq_gcs_key.json"
    )
