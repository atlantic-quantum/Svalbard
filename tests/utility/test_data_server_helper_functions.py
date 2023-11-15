import pytest
from pymongo.collection import Collection
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
    assert "compiler_data" in document
    assert "compiled" in document["compiler_data"].keys()
    assert "code" not in document["compiler_data"]["compiled"]

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
    assert "compiler_data" in document
    assert "compiled" in document["compiler_data"]
    assert "code" not in document["compiler_data"]["compiled"]
    assert "name" in document
    assert "date" in document
    assert "data_path" in document

    document = ds_helper.get_document(mdb_helper_config, result.inserted_id)
    assert isinstance(document, dict)
    assert "instruments" in document
    assert "code" in document["compiler_data"]["compiled"]


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
