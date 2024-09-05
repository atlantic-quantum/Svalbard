"""Module that defines Metadata Backends for the DataSever"""

from .mongodb_backend import MongoDBBackend

REGISTERED_METADATA_BACKENDS = {"MONGODB": MongoDBBackend}
