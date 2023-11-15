"""MongoDB Metadata Backend"""
import json
import logging
import os
from pathlib import Path

from bson.errors import InvalidId
from bson.objectid import ObjectId
from motor.core import AgnosticCollection, AgnosticDatabase
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.results import InsertOneResult
from pymongo.server_api import ServerApi

from ...data_model.data_file import BaseMetaData, MetaDataDiscriminator
from ...data_model.ipc import MeasurementHandle
from ...utility.bson import get_codec_options
from ...utility.logger import logger
from ..errors import StreamAlreadyPreparedError, StreamNotPreparedError
from .abstract_metadata_backend import (
    AbstractMetadataBackend,
    AbstractMetadataBackendConfig,
)

# ! currently metadata that was initialized cant be loaded
# ! as we cant create a MetaData object from an empty dictionary


class MongoDBBackend(AbstractMetadataBackend):
    """MongoDB Metadata Backend"""

    def __init__(
        self, url: str, database: str, collection: str, certificate: str = ""
    ) -> None:
        self._url = url
        if certificate:
            self.client = AsyncIOMotorClient(  # pragma: no cover
                self._url,
                tls=True,
                tlsCertificateKeyFile=certificate,
                server_api=ServerApi("1"),
            )
            # testing this would require  giving the jenkins server or anyone running
            # the tests a certificate (consitently located or the same certificate)
            # this is not desirable as it would decrease the security of the server
        else:
            self.client = AsyncIOMotorClient(self._url)
        self._database: AgnosticDatabase = self.client.get_database(
            database,
        )  # type: ignore
        self._collection: AgnosticCollection = self._database.get_collection(
            collection, codec_options=get_codec_options()
        )  # type: ignore
        self._logger = logger  # logging.getLogger("MongoDB_Backend")
        self._logger.setLevel(logging.DEBUG)
        self._handles = {}

    @property
    def url(self):
        """The URL of the Backend"""
        return self._url

    @property
    def database(self) -> AgnosticDatabase:
        """The MongoDB database the Backend is connected to"""
        return self._database

    @property
    def collection(self) -> AgnosticCollection:
        """The MongoDB collection the Backend is connected to"""
        return self._collection

    @property
    def handles(self):
        """A dict relating handles to paths"""
        return self._handles

    @property
    def logger(self) -> logging.Logger:
        """logger for the metadata backend"""
        return self._logger

    async def _save(self, dictionary: dict) -> Path:
        """Saves a dictionary to a MongoDB Database and returns the ObjectId of the
        inserted dictionary as path

        Args:
            dictionary (dict): the dictionary to save

        Returns:
            Path: the object id of the saved dictionary as Path
        """
        result: InsertOneResult = await self.collection.insert_one(dictionary)
        oid: ObjectId = result.inserted_id
        self.logger.debug("Metadata saved with ObjectID: %s", oid)
        return self.objectid_to_path(oid)

    async def save(self, metadata: BaseMetaData) -> Path:
        # todo AQC-289 - remove hack
        dictionary = json.loads(metadata.json())
        if os.name == "nt" and dictionary["data_path"]:  # pragma: no cover
            dictionary["data_path"] = str(dictionary["data_path"]).replace("\\", "/")
        return await self._save(dictionary)

    async def update(self, path: Path, metadata: BaseMetaData):
        dictionary = json.loads(metadata.json())
        if os.name == "nt" and dictionary["data_path"]:  # pragma: no cover
            dictionary["data_path"] = str(dictionary["data_path"]).replace("\\", "/")
        result: dict = await self.collection.find_one_and_replace(
            self.id_query(path), dictionary
        )
        oid: ObjectId = result["_id"]
        self.logger.debug("Metadata updated with ObjectID: %s", oid)
        return self.objectid_to_path(oid)

    async def load(self, path: Path) -> BaseMetaData:
        try:
            self.logger.debug("Loading Metadata with ObjectID: %s", path)
            metadata = await self.collection.find_one(self.id_query(path))
            assert metadata is not None
            del metadata["_id"]
            # todo AQC-289 - remove hack
            # hack start
            if os.name != "nt":
                # this part of the hack may be required for backwards compatibility
                # alternativly we could change the docs in the database
                if isinstance(metadata["data_path"], str):
                    metadata["data_path"] = metadata["data_path"].replace("\\", "/")
                elif isinstance(metadata["data_path"], Path):
                    metadata["data_path"] = str(metadata["data_path"]).replace(
                        "\\", "/"
                    )
            else:  # pragma: no cover
                if isinstance(metadata["data_path"], str):
                    metadata["data_path"] = metadata["data_path"].replace("/", "\\")
            # hack end
            self.logger.debug("Loaded Metadata with ObjectID: %s", path)
            return MetaDataDiscriminator(**{"metadata": metadata}).metadata
        except InvalidId as iderr:
            raise FileNotFoundError(f"No metadata found at path: {path}") from iderr

    async def init_data(self, metadata: BaseMetaData) -> Path:
        return await self._save({})

    async def prepare_stream(
        self, handle: MeasurementHandle, metadata: BaseMetaData
    ) -> Path:
        if handle.handle in self.handles:
            raise StreamAlreadyPreparedError(handle, self.__class__.__name__)
        path = await self.save(metadata)
        self.handles[handle.handle] = path
        return path

    async def finalize_stream(
        self, handle: MeasurementHandle, metadata: BaseMetaData | None = None
    ):
        if handle.handle not in self.handles:
            raise StreamNotPreparedError(handle, self.__class__.__name__)
        if metadata:
            await self.update(self.handles[handle.handle], metadata)
        # finally remove the handle from managed handles
        del self.handles[handle.handle]

    @classmethod
    def objectid_to_path(cls, oid: ObjectId) -> Path:
        """Converts a MongoDB ObjectId to pathlib Path"""
        return Path(str(oid))

    @classmethod
    def path_to_objectid(cls, path: Path) -> ObjectId:
        """Converts a pathlib Path to MongoDB ObjectId"""
        return ObjectId(str(path))

    @classmethod
    def id_query(cls, path: Path) -> dict[str, ObjectId]:
        """Converts a path to a MongoDB query"""
        return {"_id": cls.path_to_objectid(path)}


class MongoDBConfig(AbstractMetadataBackendConfig):
    """Configuration for Mongo DB Backend"""

    url: str
    database: str
    collection: str
    certificate: str = ""

    def init(self) -> MongoDBBackend:
        return MongoDBBackend(**self.dict())
