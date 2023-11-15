""" Main app for the FastAPI app that manages shared memory """
import argparse
import functools

from fastapi import FastAPI
from svalbard import launch
from svalbard.data_router import DATA_CONFIG_TEST_PATH, ds_router, lifespan

parser = argparse.ArgumentParser(
    prog="AQ Data Server", description="Launch the AQ Data Server"
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Use test collection and bucket"
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.test and DATA_CONFIG_TEST_PATH.exists():
        lifespan_ = functools.partial(
            lifespan,
            data_config_path=DATA_CONFIG_TEST_PATH,
        )
    else:
        lifespan_ = lifespan
    app = FastAPI(lifespan=lifespan_)
    app.include_router(ds_router, prefix="/data")
    launch(app, log_level="debug")
