"""Testing of MongoDB Backend"""
from pathlib import Path

import pytest
from bson.objectid import ObjectId
from svalbard.data_model.data_file import MeasurementHandle, MetaData
from svalbard.data_server.errors import (
    StreamAlreadyPreparedError,
    StreamNotPreparedError,
)
from svalbard.data_server.metadata_backend.mongodb_backend import (
    MongoDBBackend,
    MongoDBConfig,
)


@pytest.mark.asyncio
async def test_init_from_config(mdb_config: MongoDBConfig):
    """Test initialisation of a MongoDBBackend from MongoDBConfig"""
    c_mdb_backend = mdb_config.init()
    assert mdb_config.url == c_mdb_backend.url
    assert mdb_config.database == c_mdb_backend.database.name
    assert mdb_config.collection == c_mdb_backend.collection.name


@pytest.mark.asyncio
async def test_end_to_end_save_load(server_address):
    """Full end to end saving and loading test
    with everything implemented in the test itself and not fixtures"""
    # Create metadata file.
    mdata = MetaData(name="test_metadata")

    # setup backend connected to mongo db server
    # (getting server address as fixture, determined by location)
    mdb_backend = MongoDBBackend(
        f"mongodb://root:example@{server_address}:27017",
        "aq_test",
        "test",
    )

    # save the metadata
    path = await mdb_backend.save(metadata=mdata)
    assert isinstance(path, Path)

    # load the metadata back
    l_mdata = await mdb_backend.load(path)
    assert l_mdata == mdata


@pytest.mark.asyncio
async def test_save_load(mdb_backend: MongoDBBackend, metadata: MetaData):
    """test saveing and loading metadata using a MongoDB database"""
    # test saving and loading
    path = await mdb_backend.save(metadata)
    assert isinstance(path, Path)
    l_metadata = await mdb_backend.load(path)
    assert l_metadata == metadata

    # test converting path to ObjectID and back
    oid = MongoDBBackend.path_to_objectid(path)
    assert isinstance(oid, ObjectId)
    assert MongoDBBackend.objectid_to_path(oid) == path


@pytest.mark.asyncio
async def test_loading_non_existent_file(mdb_backend: MongoDBBackend):
    """test that loading a file that does not exist raises a FileNotFoundError"""
    # test saving and loading
    with pytest.raises(FileNotFoundError):
        await mdb_backend.load(Path("/this/does/not/exist"))


@pytest.mark.asyncio
async def test_init_update(mdb_backend: MongoDBBackend, metadata: MetaData):
    """Test initializing empty data and then updating it"""
    path = await mdb_backend.init_data(metadata)
    assert isinstance(path, Path)
    no_data = await mdb_backend.collection.find_one(mdb_backend.id_query(path))
    assert no_data == mdb_backend.id_query(path)

    new_path = await mdb_backend.update(path, metadata)
    assert new_path == path
    l_metadata = await mdb_backend.load(new_path)
    assert l_metadata == metadata


@pytest.mark.asyncio
async def test_stream(mdb_backend: MongoDBBackend, metadata: MetaData):
    """test perparing and finalizing stream"""
    # test preparing stream
    mh1 = MeasurementHandle.new()
    path = await mdb_backend.prepare_stream(mh1, metadata)
    assert mh1.handle in mdb_backend.handles
    assert mdb_backend.handles[mh1.handle] == path
    l_metadata = await mdb_backend.load(path)
    assert l_metadata == metadata

    # test finalising stream
    await mdb_backend.finalize_stream(mh1)
    assert mh1.handle not in mdb_backend.handles

    # test preparing stream and finalising it with new metadata
    mh2 = MeasurementHandle.new()
    path2 = await mdb_backend.prepare_stream(mh2, metadata)
    n_metadata = MetaData(name="new_name")
    await mdb_backend.finalize_stream(mh2, metadata=n_metadata)
    ln_metadata = await mdb_backend.load(path2)
    assert ln_metadata != metadata
    assert ln_metadata == n_metadata

    # test that finalizing a measurement handle
    # that has not been prepared raises StreamNotPreparedError
    mh3 = MeasurementHandle.new()
    with pytest.raises(StreamNotPreparedError):
        await mdb_backend.finalize_stream(mh3)


@pytest.mark.asyncio
async def test_stream_prepare_twice_error(
    mdb_backend: MongoDBBackend, metadata: MetaData
):
    """Test that preparing streams with same handle rasies a
    StreamAlreadyPreparedError"""
    mh1 = MeasurementHandle.new()
    await mdb_backend.prepare_stream(mh1, metadata)
    with pytest.raises(StreamAlreadyPreparedError):
        await mdb_backend.prepare_stream(mh1, metadata)
