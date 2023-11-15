"""function for launching a fastapi app and
removing shared memory from resource_tracker is needed"""
import os
from multiprocessing import resource_tracker

import colorama

colorama.init()
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

_USE_POSIX = False if os.name == "nt" else True


ACCESS_LOG_FMT = (
    "%(asctime)s - %(levelprefix)s %(client_addr)s"
    + ' - "%(request_line)s" %(status_code)s'
)
DEFAULT_LOG_FMT = "%(asctime)s - %(levelprefix)s %(message)s"


# https://github.com/python/cpython/issues/82300
def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """
    # pylint: disable=protected-access
    # pylint: disable=inconsistent-return-statements

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:  # type: ignore
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]  # type: ignore

    # pylint: enable=protected-access
    # pylint: enable=inconsistent-return-statements


def launch(
    app: FastAPI,
    address: str = "127.0.0.1",
    port: int = 5000,
    log_level: str = "info",
):
    """function for launchin a FastAPI app

    Args:
        app (FastAPI): the app to launch
        address (str, optional): ip address to launch the app at.
            Defaults to "127.0.0.1".
        port (int, optional): the port for the ip address.
            Defaults to 5000.
        log_level (str, optional): uvicorn log level.
            Defaults to "info".
        data (bool, optional):
            if true removed shared memory from the python resource tracker,
            due to a bug if this is not done memory can be garbage collected
            before it should be, this should only be done for the app that is
            responsible for creating the shared memory.
            Defaults to False.
    """

    if _USE_POSIX:
        remove_shm_from_resource_tracker()
    log_config = uvicorn.config.LOGGING_CONFIG  # type: ignore
    log_config["disable_existing_loggers"] = False
    log_config["formatters"]["access"]["fmt"] = ACCESS_LOG_FMT
    log_config["formatters"]["default"]["fmt"] = DEFAULT_LOG_FMT
    uvicorn.run(
        app,
        host=address,
        port=port,
        log_level=log_level,
        log_config=log_config,
    )
