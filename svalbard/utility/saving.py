"""
Standalone convenience functions for saving data to a running data server from
a separate process.
"""

import asyncio
import warnings
from pathlib import Path

import httpx

from ..data_model.data_file import DataFile
from ..data_model.ipc import SavedPath
from ..launch import DEFAULT_SERVER_URL


def _check_save_response(response: httpx.Response) -> None:
    """
    Checks the response from the data server and raises an exception if the
    response indicates an error.

    Args:
        response (httpx.Response):
            Response from the data server

    Raises:
        httpx.HTTPStatusError:
            If the response status code is not 201
    """
    if response.status_code != 201:
        raise httpx.HTTPStatusError(
            f"Error saving data file: {response.status_code}",
            request=response.request,
            response=response,
        )


def save_data_file(
    datafile: DataFile, data_server_url: str = DEFAULT_SERVER_URL, timeout: float = 10.0
) -> Path:
    """
    Connects to the data server at data_server_url
    and uses it to save the datafile to the configured database synchronously

    Args:
        datafile (DataFile):
            DataFile object to be saved
        data_server_url (str):
            url of the data server, defaults to http://127.0.0.1:5000
        timeout (float):
            timeout for the http request, defaults to 10.0 seconds

    Returns:
        (Path):
            containing the ObjectID string of the saved data
    """
    response = httpx.post(
        f"{data_server_url}/data/save",
        content=datafile.model_dump_json(),
        timeout=timeout,
    )
    _check_save_response(response)
    return SavedPath(**response.json()).path


async def save_data_file_async(
    datafile: DataFile, data_server_url: str = DEFAULT_SERVER_URL, timeout: float = 10.0
) -> Path:
    """
    Connects to the data server at data_server_url
    and uses it to save the datafile to the configured database asynchronously

    Args:
        datafile (DataFile):
            DataFile object to be saved
        data_server_url (str):
            url of the data server, defaults to http://127.0.0.1:5000
        timeout (float):
            timeout for the http request, defaults to 10.0 seconds

    Returns:
        (Path):
            containing the ObjectID string of the saved data
    """
    async with httpx.AsyncClient(base_url=data_server_url, timeout=timeout) as client:
        response = await client.post("/data/save", content=datafile.model_dump_json())
        _check_save_response(response)
        return SavedPath(**response.json()).path


def save_data_file_in_background(
    datafile: DataFile, data_server_url: str = DEFAULT_SERVER_URL, timeout: float = 10.0
) -> asyncio.Task:
    """
    Wrapper around save_data_file_async that runs the async function in a separate
    thread, that allows the function to be called from a synchronous context.

    Args:
        datafile (DataFile):
            DataFile object to be saved
        data_server_url (str):
            url of the data server, defaults to http://127.0.0.1:5000
        timeout (float):
            timeout for the http request, defaults to 10.0 seconds

    Returns:
        (asyncio.Task):
            For the saving task. after the task has been completed the result can be
            accessed using task.result(), the expected result is the path containing the
            ObjectID string of the saved data
    """

    def print_after(task: asyncio.Task):
        if task.exception() is not None:
            try:
                print(task.exception())
                task.result()  # calling result() will raise the exception
            except httpx.ReadTimeout:
                # timeout exceptions do not seem to cause issues
                # with actually saving the data on the data server.
                # We currently just print a warning and continue.
                #
                # The intent is to allow the user to determine if
                # any data was lost due to the timeout.
                warnings.warn(
                    "Timeout exception raised while saving data file to data server.",
                    UserWarning,
                )

    # handle calling from jupyter notebook
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover
        loop = asyncio.get_event_loop()
    save_task = loop.create_task(
        save_data_file_async(datafile, data_server_url, timeout)
    )
    save_task.add_done_callback(print_after)
    return save_task
