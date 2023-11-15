"""
Custom codecs for Mongo DB BSON encoding
"""

import os
import pathlib
from typing import Any, Mapping, MutableMapping, TypeVar, Union

from bson import BSON
from bson.binary import USER_DEFINED_SUBTYPE, Binary
from bson.codec_options import CodecOptions, TypeCodec, TypeRegistry
from bson.raw_bson import RawBSONDocument

_DocumentIn = Union[MutableMapping[str, Any], "RawBSONDocument"]
_DocumentType = TypeVar("_DocumentType", bound=Mapping[str, Any])


class PosixPathCodec(TypeCodec):
    """Custom Codec to convert PosixPaths to str when encoding to BSON"""

    # adapted from https://stackoverflow.com/questions/6367589/
    python_type = (
        pathlib.WindowsPath if os.name == "nt" else pathlib.PosixPath  # type: ignore
    )
    bson_type = Binary  # type: ignore

    def transform_python(self, value):
        return Binary(str(value).encode("UTF-8"), subtype=USER_DEFINED_SUBTYPE)

    def transform_bson(self, value):
        if value.subtype == USER_DEFINED_SUBTYPE:
            return pathlib.Path(value.decode("UTF-8"))
        return value


def get_codec_options() -> CodecOptions:
    """function to construct a BSON CodecOptions using the NumpyCodec

    Returns:
        CodecOptions: BSON CodecOptions
    """
    type_registry = TypeRegistry([PosixPathCodec()])
    return CodecOptions(type_registry=type_registry)


def encode(doc: _DocumentIn) -> BSON:
    """wrap BSON.encode with our custom encoding"""
    return BSON.encode(doc, codec_options=get_codec_options())


def decode(bson: BSON) -> _DocumentType:
    """wrap BSON.decode with our custom encoding"""
    return BSON.decode(bson, codec_options=get_codec_options())
