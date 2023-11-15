# Release Notes

## 0.0.8
---
 - AQC-263: Changed name from aq_data to svalbard, prepared for moving repository to open source.

## 0.0.7
---
 - Bumped python requirement to 3.10 and version to 0.0.7
 - AQC-579: Added methods for updating data and metadata after initialization, updating data support partial update.
 - AQC-577: Data Server app now attempts using config from .aq_configs to set itself up during startup

## 0.0.6
---
 - AQC-407: Changed favorites field in MetaData to Flags (enumeration), meta data version bumped to 0.1.2
 - AQC-432: Feature: Synchronous Frontend V1.
 - AQC-407: Feature: Added favorites field to MetaData.
 - AQC-403: Feature: Added function to create a Data object from a Measurement object.
 - AQC-404: Feature: Added helper functions to interface with the data server.
 - AQC-327: Feature: Added support for saving arb. files to the data router.
 - AQC-325: Feature: Added support for locations of arb. files to metadata model.
 - AQC-324: Feature: Added support for arb. files in data model.
 - AQC-359: Feature: Added Measurement data model.
 - AQC-323: Feature: Added support for saving/loading arb. files in data backend.
 - AQC-576: Bugfix: Added colorama as dependency

## 0.0.5
---
 - AQC-291: Feature: Zarr Metadata of FS Backend is now consolidated sacrificing slightly longer saveing for faster loading.
 - AQC-277: Fixed: Loading files on unix that where saved of windows.
 - AQC-278: Fixed: datetime generation for metadata.
 - AQC-253: Add functions to retrieve partial data loading. (Added SliceListModel and SliceModel types and load_partial functions in data router, frontend, and backend).

## 0.0.4
---
 - AQC-120: Defined Structure of DataFile (large changes to MetaData).
 - AQC-266: Improved validator of InstrumentSetting to handle initialization without type.

## 0.0.3
---
 - Added Release Notes.
 - Improved initialization of FrontentV1 to accuretly create FS or GCS data backend.
 - AQC-246 Added option to use X509 credentials to access metadata backend. Changed remote demo to use mongoDB Atlas instead of local Dockerized mongo server.

## 0.0.2
---
 - AQC-234 Added Close methods to web api.
 - AQC-232 Updated to work with windows, added rudimentary management of shared memories.

## 0.0.1
---
 - AQC-184 Test Building with python 3.11.
 - AQC-155 Data Server Sprint review demo.
 - AQC-154 Docker Compose file for CI/CD.
 - AQC-136 datasaver front end.
 - AQC-147 change jenkinsfile in aq_data to streamline testing.
 - AQC-139 design process of streaming data.
 - AQC-138 datasaver backend for data.
 - AQC-137 datasaver backend for metadata.