[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Atlantic Quantum Data

This package contains the data server solution and data models used by Atlantic Quantum, it can be used by itself to store and retrive data but is also integrated into Atlantic Quantum's larger measurement framework. 

The central concept for data storage in this package is the `DataFile` which has two components, `DataFile.data`, which contains large arrays of raw or processed measurement data, and `DataFile.metadata`, which contains all the information about the `data` such as when it was measured, all instrument settings during measurements, qasm scripts used to configure the measurement etc. 

These two components are stored in different locations. The `metadata` is stored in a database, such as MongoDB, and is searchable using database queries while the `data` is stored in blob storage such as Google Cloud Storage. The precise backends used to store / access the `data` and `metadata` are user configureable. Currently filesystem storage and GCS are supported for `data` storage and MongoDB is supported for `metadata` storage.

## Status

This package is stable, well tested and used daily to save / access all of Atlantic Quantum's quantum measurements.

## Setup

The app used to run the data server is configured based on the file `~/.aq_config/data_server.json`. Two setup examples can be found in the examples folder, `examples/data_server.json` and `examples/data_server_local.json`.

The `examples/data_server_local.json` file can be used as is and is configured to use the dockerized mongoDB instance (described in the Testing section below) for metadata storage and an in memory file system for data storage. These storage methods make this confiuration suitable for testing purposes but unsuitable for use in production.

The `examples/data_server.json` file needs configuration by the user before it can be used. the `metadata_backend.url`, `metadata_backend.database` and `metadata_backend.collection` should be configured to point to actual mongoDB instance and a database+collection on that database instance, a path to an X509 certificate can optionally be provided for authentication. For `data_backend` configuration, system wide default `gcloud` credentials are used for authentication, but GCS bucket name `data_backend.path` and google cloud project name `data_backend.project` need to be provided.

## Usage

Once the required configuration files have been created and placed in the correct locations the data server app can be launched by running the launch file found in the examples folder, i.e.

```
python examples/launch.py
```

With default settings documentation of API commands supported by the data server app supports should be accessible at `http://127.0.0.1:5000/docs` once the app has been launched. The `examples/demo.py` file demonstrates the basic usage of the api to save and load data.

## Testing

Testing this module requires running both a MongoDB instance and a fake Google Cloud Storage Server.

These services can be launched by using the included 'docker-compose.yml' file, 
make sure you have [docker desktop](https://www.docker.com/products/docker-desktop/) installed
and then run the following command in a terminal window open in the repository directory.
```
docker-compose up -d
```
Once the Docker containers are running the tests can be executed using
```
pytest tests
```