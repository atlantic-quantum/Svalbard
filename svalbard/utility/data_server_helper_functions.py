"""
Helper functions for retrieving metadata from the mongodb database.

These functions are meant to be used standalone, and not in well integrated
production code. E.g. each function opens a new connection to the database
and does not close it. This is done to make the functions as simple as possible.

this module implements a synchronous version of the functions, it is possible
to implement an asynchronous version of the functions using the motor library.

"""


from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.server_api import ServerApi

from ..data_server.metadata_backend.mongodb_backend import MongoDBConfig


def get_collection(config: MongoDBConfig) -> Collection:
    """
    Get the collection from a MongoDBConfig object.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.

    Returns:
        Collection: The collection pointed to by the MongoDBConfig object.
    """
    client = MongoClient(
        config.url,
        tlsCertificateKeyFile=config.certificate,
        server_api=ServerApi("1"),
    )
    database = client.get_database(config.database)
    return database.get_collection(config.collection)


def get_number_of_documents(config: MongoDBConfig, query: dict | None = None) -> int:
    """
    Get the number of documents in a collection matching a query.
    an empty query will return the total number of documents in the collection.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        query (dict | None, optional):
            A query to filter the documents. Defaults to None. which will
            return the total number of documents in the collection.
    """
    query = query or {}
    return get_collection(config).count_documents(query)


def get_document(
    config: MongoDBConfig, object_id: str, projection: dict | None = None
) -> dict:
    """
    Get a document from a collection by its object id. An optional projection can be
    used, only the fields indicated in the projection will be included in the returned
    document.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        object_id (str):
            Object id of the document to retrieve.
        projection (dict | None, optional):
            the fields of the returned documents are projected onto this dict.
            Defaults to None, which returns all the values in the document.

            e.g. projection = {"_id": 0, "name": 1} will return the name field of
            the document and exclude the _id field (and all other fields).

            if all fields in the projection are 0, then those fields will be excluded
            and all other fields will be included. e.g.
            projection = {"instruments": 0, "compiler_data.compiled.code": 0}

    Returns:
        dict: document from the collection.
    """
    document = get_collection(config).find_one(
        {"_id": ObjectId(object_id)}, projection=projection
    )
    assert isinstance(document, dict)
    return document


def get_name_and_date(config: MongoDBConfig, object_id: str) -> dict:
    """
    Get the name and date of a document from a collection by its object id.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        object_id (str):
            object_id of the document to retrieve.

    Returns:
        dict: document from the collection with only the name and date fields.
    """
    return get_document(config, object_id, {"_id": 1, "name": 1, "date": 1})


def get_and_exclude_large_fields(config: MongoDBConfig, object_id: str) -> dict:
    """
    Get a document from a collection by its object id. Exclude the instruments and
    and compiled code fields.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        object_id (str):
            object_id of the document to retrieve.
    Returns:
        dict:
            document from the collection with the instruments and compiled code fields
            excluded.

    """
    return get_document(
        config, object_id, {"instruments": 0, "compiler_data.compiled.code": 0}
    )


def get_name_and_data_path(config: MongoDBConfig, object_id: str) -> dict:
    """
    Get the name and data path of a document from a collection by its object id.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        object_id (str):
            object_id of the document to retrieve.

    Returns:
        dict: document from the collection with only the name and data_path fields.
    """
    return get_document(config, object_id, {"_id": 1, "name": 1, "data_path": 1})


def get_many_documents(
    config: MongoDBConfig,
    filter: dict,
    projection: dict | None = None,
) -> Cursor:
    """
    Return a cursor to documents in a collection matching a filter. An optional
    projection can be used, only the fields indicated in the projection will be
    returned. (except if all fields in the projection are 0, then those fields will
    not be returned and all other fields will be returned).

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        filter (dict):
            A query to filter the documents.
        projection (dict | None, optional):
            the fields of the returned documents are projected onto this dict.
            Defaults to None, which returns all the values in the document.

            e.g. projection = {"_id": 0, "name": 1} will return the name field of
            the document and exclude the _id field (and all other fields).

            if all fields in the projection are 0, then those fields will be excluded
            and all other fields will be included. e.g.
            projection = {"instruments": 0, "compiler_data.compiled.code": 0}

    Returns:
        Cursor:
            an iterable cursor to the documents in the collection matching the filter
            query and projected onto the projection dict.

            to retrieve the documents on can iterate over the cursor or access them by
            index

            e.g.

            cursor = get_many_documents(config, filter, projection)
            documents = cursor[0] <- returns the first document
            documents = cursor[0:10] <- returns the first 10 documents

            Note: the documents are not retrieved from the database until the cursor is
            actually iterated over of accessed.
    """
    return get_collection(config).find(filter, projection=projection)


def get_name_and_date_all(config: MongoDBConfig) -> Cursor:
    """
    returns a cursor to all the documents in the collection with only the name and date
    of each document.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.

    Returns:
        Cursor:
            An iterable cursor to the documents in the collection.
    """
    return get_many_documents(config, {}, {"_id": 1, "name": 1, "date": 1})


def update_document(config: MongoDBConfig, object_id: str, update: dict) -> None:
    """
    Update a document in a collection by its object id.

    Args:
        config (MongoDBConfig):
            A MongoDBConfig object containing the information to connect to
            the database.
        object_id (str):
            object_id of the document to update.
        update (dict):
            the update to apply to the document.
    """
    get_collection(config).update_one({"_id": ObjectId(object_id)}, update)
