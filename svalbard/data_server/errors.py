"""Custom Errors for the Data Server"""

from ..data_model.ipc import MeasurementHandle
from ..utility.logger import logger


class BaseStreamError(Exception):
    """Base class for stream errors"""

    _DEFAULT_MESSAGE = "stream error message"

    def __init__(
        self,
        handle: MeasurementHandle,
        clss: str,
        msg: str = "",
    ) -> None:
        msg = msg if msg else self._DEFAULT_MESSAGE
        self.msg = f"{clss}: {msg} {handle.handle}"
        logger.error(self.msg)

    def __str__(self) -> str:
        return self.msg


class StreamAlreadyPreparedError(BaseStreamError):
    """Error raised if trying to prepare a stream
    with a handle that a stream has already been prepared for"""

    _DEFAULT_MESSAGE = "Already prepared stream with MeasurmentHandle:"


class StreamNotPreparedError(BaseStreamError):
    """Error raised if trying to finalize or save a buffer
    using a handle that a stream has not been prepared for"""

    _DEFAULT_MESSAGE = "Stream has not been prepared using MeasurmentHandle:"


class BufferShapeError(Exception):
    """Error Raised if buffer refrence, supplied when attempting
    to save a buffer, does not have the same shape as the data
    the stream was prepared for"""


class DataNotInitializedError(BaseStreamError):
    """Error raised if trying to update data that has not been initialized"""

    _DEFAULT_MESSAGE = "Data has not been initialized for MeasurmentHandle:"
