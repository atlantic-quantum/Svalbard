"""Test custom BSON codec"""

import os
import uuid
from pathlib import Path, PosixPath, WindowsPath

import pytest
from bson import BSON
from bson.binary import Binary, UuidRepresentation
from bson.codec_options import CodecOptions, TypeRegistry
from bson.errors import InvalidDocument

from svalbard.utility.bson import PosixPathCodec, decode, encode


@pytest.fixture(name="posix_path_dict")
def fixture_posix_path_dict():
    """fixture that creates dictionary with PosixPath value"""
    yield {"a": Path("/tmp/test/posix/path/dict"), "c": "other/value"}


@pytest.fixture(name="regular_dict")
def fixture_regular_dict():
    """fixture with a regular string"""
    yield {"b": "/tmp/test/regular/string/dict"}


@pytest.fixture(name="with_uuid")
def fixture_with_uuid(posix_path_dict):
    """fixture with uuid to test encoding other binary values a the same time"""
    posix_path_dict["d"] = uuid.uuid4()
    yield posix_path_dict


def test_encode(posix_path_dict):
    """Test that using our codec we can encode and decode
    PosixPaths to bson documents"""
    with pytest.raises(InvalidDocument):
        BSON.encode(posix_path_dict)
    bson_doc = encode(posix_path_dict)
    assert isinstance(bson_doc, BSON)


def test_decode(posix_path_dict):
    """Test that using our codec we can encode and decode
    PosixPaths to bson documents"""
    bson_doc = encode(posix_path_dict)

    decoded_bson = BSON.decode(bson_doc)
    assert isinstance(decoded_bson, dict)
    assert "a" in decoded_bson
    assert isinstance(decoded_bson["a"], Binary)

    m_decoded_bson = decode(bson_doc)
    assert isinstance(m_decoded_bson, dict)
    assert "a" in m_decoded_bson
    path_type = WindowsPath if os.name == "nt" else PosixPath
    assert isinstance(m_decoded_bson["a"], path_type)

    assert posix_path_dict == m_decoded_bson
    assert m_decoded_bson != decoded_bson


def test_regular_strings_give_same_result(regular_dict):
    """Test that the standard codec and our custom codec handle
    regular strings the same way"""
    standard_bson = BSON.encode(regular_dict)
    custom_bson = encode(regular_dict)
    assert standard_bson == custom_bson

    standard_decoded = BSON.decode(standard_bson)
    custom_decoded = decode(custom_bson)
    assert standard_decoded == custom_decoded
    assert standard_decoded == regular_dict
    assert custom_decoded == regular_dict


def test_other_binary_not_decoded_to_posix_path(with_uuid):
    """Test that other binary values don't get decoded to posix path"""
    std_opts = CodecOptions(
        uuid_representation=UuidRepresentation.STANDARD,
        type_registry=TypeRegistry([PosixPathCodec()]),
    )
    bson_doc = BSON.encode(with_uuid, codec_options=std_opts)
    decode(bson_doc)
