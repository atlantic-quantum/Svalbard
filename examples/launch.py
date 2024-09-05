""" Main app for the FastAPI app that manages shared memory """

import argparse
import functools

import fastapi
import setproctitle
from fastapi import FastAPI

from svalbard import launch
from svalbard.data_router import (
    DATA_CONFIG_FILE_PATH,
    DATA_CONFIG_LOCAL_PATH,
    DATA_CONFIG_PATH,
    DATA_CONFIG_TEST_PATH,
    ds_router,
    lifespan,
)
from svalbard.launch import DEFAULT_SERVER_PORT

parser = argparse.ArgumentParser(
    prog="AQ Data Server", description="Launch the AQ Data Server"
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Use test collection and bucket"
)
parser.add_argument(
    "-l", "--local", action="store_true", help="Use local mongodb + Fake GCS"
)
parser.add_argument(
    "-f", "--file", action="store_true", help="Use local mongodb + local file system"
)
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=DEFAULT_SERVER_PORT,
    help=f"Port to run the server on (default: {DEFAULT_SERVER_PORT})",
)

DATA_SERVER_CONFIGS = {
    "test": DATA_CONFIG_TEST_PATH,
    "local": DATA_CONFIG_LOCAL_PATH,
    "file": DATA_CONFIG_FILE_PATH,
}


def get_data_config_path(args):
    for key, path in DATA_SERVER_CONFIGS.items():
        if args.get(key) and path.exists():
            return path
    return DATA_CONFIG_PATH


if __name__ == "__main__":
    setproctitle.setproctitle("svalbard_server")
    args = parser.parse_args()
    config_path = get_data_config_path(vars(args))
    lifespan_ = functools.partial(lifespan, data_config_path=config_path)
    app = FastAPI(
        lifespan=lifespan_, default_response_class=fastapi.responses.ORJSONResponse
    )
    app.include_router(ds_router, prefix="/data")
    launch(app, log_level="debug", port=args.port)
