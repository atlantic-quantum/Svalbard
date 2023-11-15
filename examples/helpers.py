import json

from svalbard.data_router import DATA_CONFIG_PATH
from svalbard.data_server.metadata_backend.mongodb_backend import MongoDBConfig
from svalbard.utility.data_server_helper_functions import (
    get_many_documents,
    get_number_of_documents,
)

with DATA_CONFIG_PATH.open("r") as f:
    config = MongoDBConfig(**json.load(f)["metadata_backend"])

    print(get_number_of_documents(config))
    print(get_many_documents(config, {}, projection={"_id": 1, "name": 1})[0])
