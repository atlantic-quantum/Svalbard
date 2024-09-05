import json
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from bson import ObjectId
from pymongo.collection import Collection

from svalbard.data_model.data_file import DataFile
from svalbard.data_server.frontend.frontend_v1 import FrontendV1Config
from svalbard.data_server.frontend.sync_frontend_v1 import SyncFrontendV1
from svalbard.data_server.metadata_backend.mongodb_backend import MongoDBConfig
from svalbard.utility import data_server_helper_functions as ds_helper


@pytest.fixture(name="mdb_helper_config")
def fixture_mdb_helper_config(mdb_config: MongoDBConfig):
    mdb_config.certificate = None
    collection = ds_helper.get_collection(mdb_config)
    collection.insert_one(
        {
            "name": "test1",
            "date": "test1",
            "data_path": "test1",
            "instruments": {"test1": {"identity": "test1"}},
            "compiler_data": {"compiled": {"code": "test1", "identity": "test1"}},
        }
    )
    yield mdb_config
    collection.drop()


@pytest.fixture(name="oid")
def fixture_oid(mdb_config: MongoDBConfig):
    cursor = ds_helper.get_many_documents(
        mdb_config,
        {
            "name": {"$exists": True},
            "date": {"$exists": True},
            "data_path": {"$exists": True},
            "instruments": {"$exists": True},
            "compiler_data": {"$exists": True},
        },
    )
    document = cursor[0]
    yield document["_id"]


def test_get_collection(mdb_helper_config: MongoDBConfig):
    """Test that get_collection returns a collection"""
    collection = ds_helper.get_collection(mdb_helper_config)
    assert isinstance(collection, Collection)


def test_get_collection_from_path(mdb_helper_config: MongoDBConfig):
    """Test that get_collection returns a collection when config is a path"""
    with NamedTemporaryFile("w", suffix=".json") as f:
        f.write(mdb_helper_config.model_dump_json())
        f.seek(0)
        collection = ds_helper.get_collection(f.name)
    assert isinstance(collection, Collection)


def test_get_collection_from_path_with_metadata_backend(
    mdb_helper_config: MongoDBConfig,
):
    """
    Test that get_collection returns a collection when config is a path and
    the file contains a metadata_backend field
    """
    with NamedTemporaryFile("w", suffix=".json") as f:
        model_dict = {"metadata_backend": mdb_helper_config.model_dump()}
        f.write(json.dumps(model_dict))
        f.seek(0)
        collection = ds_helper.get_collection(f.name)
    assert isinstance(collection, Collection)


def test_get_number_of_documents(mdb_helper_config: MongoDBConfig):
    """Test that get_number_of_documents returns an int"""
    n_docs = ds_helper.get_number_of_documents(mdb_helper_config)
    assert isinstance(n_docs, int)


def test_get_document(mdb_helper_config: MongoDBConfig, oid: str):
    """Test that get_document returns a dict"""
    document = ds_helper.get_document(mdb_helper_config, oid)
    assert isinstance(document, dict)


def test_get_many_documents(mdb_helper_config: MongoDBConfig):
    """Test that get_many_documents returns a list"""
    cursor = ds_helper.get_many_documents(mdb_helper_config, {})
    assert isinstance(cursor[0], dict)


def test_get_name_and_date(mdb_helper_config: MongoDBConfig, oid: str):
    """Test that get_name_and_date returns a dict with name and date fields"""
    document = ds_helper.get_name_and_date(mdb_helper_config, oid)
    assert isinstance(document, dict)
    assert "name" in document
    assert "date" in document


def test_get_name_and_data_path(mdb_helper_config: MongoDBConfig, oid: str):
    """Test that get_name_and_data_path returns a dict with name and data path fields"""
    document = ds_helper.get_name_and_data_path(mdb_helper_config, oid)
    assert isinstance(document, dict)
    assert "name" in document
    assert "data_path" in document


def test_get_and_exclude_large_fields(mdb_helper_config: MongoDBConfig, oid: str):
    """
    Test that get_and_exclude_large_fields returns a dict with instruments and
    compiled code fields excluded
    """
    document = ds_helper.get_and_exclude_large_fields(mdb_helper_config, oid)
    assert isinstance(document, dict)
    assert "instruments" not in document
    assert "compiler_data" not in document

    collection = ds_helper.get_collection(mdb_helper_config)
    result = collection.insert_one(
        {
            "instruments": "test",
            "compiler_data": {"compiled": {"code": "test"}},
            "name": "test",
            "date": "test",
            "data_path": "test",
        }
    )

    document = ds_helper.get_and_exclude_large_fields(
        mdb_helper_config, result.inserted_id
    )

    assert isinstance(document, dict)
    assert "instruments" not in document
    assert "compiler_data" not in document
    assert "name" in document
    assert "date" in document
    assert "data_path" in document

    document = ds_helper.get_document(mdb_helper_config, result.inserted_id)
    assert isinstance(document, dict)
    assert "instruments" in document
    assert "compiler_data" in document


def test_get_name_and_date_all(mdb_helper_config: MongoDBConfig):
    """Test that get_name_and_date returns a dict with name and date fields"""
    cursor = ds_helper.get_name_and_date_all(mdb_helper_config)
    for document in cursor[:10]:
        assert isinstance(document, dict)
        for key in document:
            assert key in ["name", "date", "_id"]


def test_update_document(mdb_helper_config: MongoDBConfig, oid: str):
    """Test that update_document updates a document"""
    document = ds_helper.get_document(mdb_helper_config, oid)
    assert document["name"] == "test1"
    ds_helper.update_document(
        mdb_helper_config, oid, {"$set": {"name": "updated_test"}}
    )
    document = ds_helper.get_document(mdb_helper_config, oid)
    assert document["name"] == "updated_test"


def test_add_field_with_update_document(mdb_helper_config: MongoDBConfig, oid: str):
    """Test that a field can be added with update_document"""
    document = ds_helper.get_document(mdb_helper_config, oid)
    assert "favorite" not in document
    ds_helper.update_document(mdb_helper_config, oid, {"$set": {"favorite": True}})
    document = ds_helper.get_document(mdb_helper_config, oid)
    assert document["favorite"]


def test_get_data_group(
    frontend_gcs_config: FrontendV1Config,
    data_file: DataFile,
    sync_frontend_mdb_gcs: SyncFrontendV1,
):
    path, _ = sync_frontend_mdb_gcs.save(data_file)
    sync_frontend_mdb_gcs.complete_background_tasks()
    print(frontend_gcs_config)
    data = ds_helper.get_data_group(frontend_gcs_config, str(path))
    for dataset in data_file.data.datasets:
        assert dataset.name in list(data.array_keys())
        assert dataset.memory.dtype == data[dataset.name].dtype
        assert dataset.memory.to_array().shape == data[dataset.name].shape
        assert np.allclose(dataset.memory.to_array(), data[dataset.name])


def test_get_many_documents_by_id(
    frontend_gcs_config: FrontendV1Config,
    data_file: DataFile,
    sync_frontend_mdb_gcs: SyncFrontendV1,
):
    """Test getting documents by id"""
    path, _ = sync_frontend_mdb_gcs.save(data_file)
    sync_frontend_mdb_gcs.complete_background_tasks()
    object_id = ObjectId(str(path))
    object_ids = [object_id]
    data = ds_helper.get_many_documents_by_id(
        frontend_gcs_config, object_ids=object_ids
    )
    assert len(data) == 1
