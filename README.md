<div align="center">
  <p>
    <a href="https://www.satellite-image-deep-learning.com/">
        <img src="logo.png" width="700">
    </a>
</p>
  <h2>Software for working with satellite & aerial imagery data & datasets.</h2>

# 👉 [satellite-image-deep-learning.com](https://www.satellite-image-deep-learning.com/) 👈

</div>

**How to use this repository:** if you know exactly what you are looking for (e.g. you have the paper name) you can `Control+F` to search for it in this page (or search in the raw markdown).

## Contents
* [Deep learning projects & frameworks](https://github.com/satellite-image-deep-learning/software#deep-learning-projects--frameworks)
* [Software for working with remote sensing data](https://github.com/satellite-image-deep-learning/software#software-for-working-with-remote-sensing-data)
* [Image dataset creation](https://github.com/satellite-image-deep-learning/software#image-dataset-creation)
* [Image chipping/tiling & merging](https://github.com/satellite-image-deep-learning/software#image-chippingtiling--merging)
* [Image processing, handling, manipulation](https://github.com/satellite-image-deep-learning/software#image-processing-handling-manipulation)
* [Image augmentation packages](https://github.com/satellite-image-deep-learning/software#image-augmentation-packages)
* [Image formats, data management and catalogues](https://github.com/satellite-image-deep-learning/software#image-formats-data-management-and-catalogues)
* [General utilities](https://github.com/satellite-image-deep-learning/software#general-utilities)
* [Low level numerical & data formats](https://github.com/satellite-image-deep-learning/software#low-level-numerical--data-formats)
* [Graphing and visualisation](https://github.com/satellite-image-deep-learning/software#graphing-and-visualisation)
* [Algorithms](https://github.com/satellite-image-deep-learning/software#algorithms)
* [GDAL & Rasterio](https://github.com/satellite-image-deep-learning/software#gdal--rasterio)
* [Cloud Optimised GeoTiff (COG)](https://github.com/satellite-image-deep-learning/software#cloud-optimised-geotiff-cog)
* [SpatioTemporal Asset Catalog specification (STAC)](https://github.com/satellite-image-deep-learning/software#spatiotemporal-asset-catalog-specification-stac)
* [OpenStreetMap](https://github.com/satellite-image-deep-learning/software#openstreetmap)
* [QGIS](https://github.com/satellite-image-deep-learning/software#qgis)
* [Parallel processing with Dask](https://github.com/satellite-image-deep-learning/software#parallel-processing-with-dask)
* [Jupyter](https://github.com/satellite-image-deep-learning/software#jupyter)
* [Julia language](https://github.com/satellite-image-deep-learning/software#julia-language)
* [Streamlit](https://github.com/satellite-image-deep-learning/software#streamlit)

## Deep learning projects & frameworks
* [TorchGeo](https://github.com/microsoft/torchgeo) -> PyTorch library providing datasets, samplers, transforms, and pre-trained models specific to geospatial data. 📺 YouTube: [TorchGeo with Caleb Robinson](https://youtu.be/ET8Hb_HqNJQ)
* [rastervision](https://rastervision.io/) -> An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets. 📺 YouTube: [Raster Vision with Adeel Hassan](https://youtu.be/hH59fQ-HhZg)
* [segmentation_gym](https://github.com/Doodleverse/segmentation_gym) -> A neural gym for training deep learning models to carry out geoscientific image segmentation, uses keras. 📺 YouTube: [Satellite image segmentation using the Doodleverse segmentation gym with Dan Buscombe](https://youtu.be/0I1TOOGfdZ0)
* [sits](https://github.com/e-sensing/sits) -> Satellite image time series in R. 📺 YouTube: [Satellite image time series with Gilberto Camara](https://youtu.be/0_wt_m6DoyI)
* [torchrs](https://github.com/isaaccorley/torchrs) -> PyTorch implementation of popular datasets and models in remote sensing
* [pytorch-enhance](https://github.com/isaaccorley/pytorch-enhance) -> Open-source Library of Image Super-Resolution Models, Datasets, and Metrics for Benchmarking or Pretrained Use
* [GeoTorchAI](https://github.com/DataSystemsLab/GeoTorchAI) -> A Deep Learning and Scalable Data Processing Framework for Raster and Spatio-Temporal Datasets, uses PyTorch and Apache Sedona
* [EarthNets](https://earthnets.nicepage.io/) -> includes a database of 400 baseline models, and tutorial examples of common deep learning tasks on satellite imagery
* [PaddleRS](https://github.com/PaddlePaddle/PaddleRS) -> remote sensing image processing development kit based on PaddlePaddle. For English see README_EN.md
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) -> Semantic Segmentation Toolbox with support for many remote sensing datasets including LoveDA, Potsdam, Vaihingen & iSAID
* [mmrotate](https://github.com/open-mmlab/mmrotate) -> Open-source toolbox for rotated object detection which is great for detecting randomly oriented objects in huge satellite images
* [Myria3D](https://github.com/IGNF/myria3d) -> Myria3D is a deep learning library designed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud.
* [Open3D-ML](https://github.com/isl-org/Open3D-ML) -> Open3D-ML focuses on applications such as semantic point cloud segmentation and provides pretrained models that can be applied to common tasks as well as pipelines for training. It works with TensorFlow and PyTorch.
* [DeepHyperX](https://github.com/eecn/Hyperspectral-Classification) -> A Python/pytorch tool to perform deep learning experiments on various hyperspectral datasets
* [DELTA](https://github.com/nasa/delta) -> Deep Earth Learning, Tools, and Analysis, by NASA is a framework for deep learning on satellite imagery, based on Tensorflow & using MLflow for tracking experiments
* [pytorch_eo](https://github.com/earthpulse/pytorch_eo) -> aims to make Deep Learning for Earth Observation data easy and accessible to real-world cases and research alike
* [NGVEO](https://github.com/ESA-PhiLab/NGVEO) -> applying convolutional neural networks (CNN) to Earth Observation (EO) data from Sentinel 1 and 2 using python and PyTorch
* [chip-n-scale-queue-arranger by developmentseed](https://github.com/developmentseed/chip-n-scale-queue-arranger) -> an orchestration pipeline for running machine learning inference at scale. [Supports fastai models](https://github.com/developmentseed/fastai-serving)
* [TorchSat](https://github.com/sshuair/torchsat) is an open-source deep learning framework for satellite imagery analysis based on PyTorch (no activity since June 2020)
* [DeepNetsForEO](https://github.com/nshaud/DeepNetsForEO) -> Uses SegNET for working on remote sensing images using deep learning (no activity since 2019)
* [RoboSat](https://github.com/mapbox/robosat) -> semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds (no longer maintained)
* [DeepOSM](https://github.com/trailbehind/DeepOSM) -> Train a deep learning net with OpenStreetMap features and satellite imagery (no activity since 2017)
* [mapwith.ai](https://mapwith.ai/) -> AI assisted mapping of roads with OpenStreetMap. Part of [Open-Mapping-At-Facebook](https://github.com/facebookmicrosites/Open-Mapping-At-Facebook)
* [terragpu](https://github.com/nasa-cisto-ai/terragpu) -> Python library to process and classify remote sensing imagery by means of GPUs and AI/ML
* [EOTorchLoader](https://github.com/ndavid/EOTorchLoader) -> Pytorch dataloader and pytorch lightning datamodule for Earth Observation imagery
* [satellighte](https://github.com/canturan10/satellighte) -> an image classification library that consist state-of-the-art deep learning methods, using PyTorch Lightning
* [rsi-semantic-segmentation](https://github.com/xdu-jjgs/rsi-semantic-segmentation) -> A unified PyTorch framework for semantic segmentation from remote sensing imagery, in pytorch, uses DeepLabV3ResNet
* [ODEON landcover](https://github.com/IGNF/odeon-landcover) -> a set of command-line tools performing semantic segmentation on remote sensing images (aerial and/or satellite) with as many layers as you wish
* [AiTLAS](https://github.com/biasvariancelabs/aitlas) -> implements state-of-the-art AI methods for exploratory and predictive analysis of satellite images
* [aitlas-arena](https://github.com/biasvariancelabs/aitlas-arena) -> An open-source benchmark framework for evaluating state-of-the-art deep learning approaches for image classification in Earth Observation (EO)
* [RocketML Deep Neural Networks](https://github.com/rocketmlhq/rmldnn) -> read [Satellite Image Classification](https://github.com/rocketmlhq/rmldnn/tree/main/tutorials/satellite_image_classification) using rmldnn and Sentinel 2 data
* [raster4ml](https://github.com/remotesensinglab/raster4ml) -> A geospatial raster processing library for machine learning
* [moonshine](https://github.com/moonshinelabs-ai/moonshine) -> a Python package that makes it easier to train models on remote sensing data like satellite imagery

## Software for working with remote sensing data
[A note on licensing](https://www.gislounge.com/businesses-using-open-source-gis/): The two general types of licenses for open source are copyleft and permissive. Copyleft requires that subsequent derived software products also carry the license forward, e.g. the GNU Public License (GNU GPLv3). For permissive, options to modify and use the code as one please are more open, e.g. MIT & Apache 2. Checkout [choosealicense.com/](https://choosealicense.com/)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code) -> lists many useful tools and resources
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)
* [dl-satellite-docker](https://github.com/sshuair/dl-satellite-docker) -> docker files for geospatial analysis, including tensorflow, pytorch, gdal, xgboost...
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [Land Cover Mapping web app from Microsoft](https://github.com/microsoft/landcover)
* [Solaris](https://github.com/CosmiQ/solaris) -> An open source ML pipeline for overhead imagery
* [openSAR](https://github.com/EarthBigData/openSAR) -> Synthetic Aperture Radar (SAR) Tools and Documents from Earth Big Data LLC
* [YMIR](https://github.com/industryessentials/ymir) -> YMIR provides a Rapid Data-centric Development Platform
for Vision Applications. Read the paper [here](https://arxiv.org/abs/2111.10046).
* [qhub](https://qhub.dev) -> QHub enables teams to build and maintain a cost effective and scalable compute/data science platform in the cloud.
* [imagej](https://imagej.net) -> a very versatile image viewer and processing program
* [Geo Data Viewer](https://github.com/RandomFractals/geo-data-viewer) extension for VSCode which enables opening and viewing various geo data formats with nice visualisations
* [Datasette](https://datasette.io/) is a tool for exploring and publishing data as an interactive website and accompanying API, with SQLite backend. Various plugins extend its functionality, for example to allow displaying geospatial info, render images (useful for thumbnails), and add user authentication. Available as a [desktop app](https://datasette.io/desktop). Read [Drawing shapes on a map to query a SpatiaLite database](https://simonwillison.net/2021/Jan/24/drawing-shapes-spatialite/)
* [Photoprism](https://github.com/photoprism/photoprism) is a privately hosted app for browsing, organizing, and sharing your photo collection, with support for tiffs
* [dbeaver](https://github.com/dbeaver/dbeaver) is a free universal database tool and SQL client with [geospatial features](https://github.com/dbeaver/dbeaver/wiki/Working-with-Spatial-GIS-data)
* [Grafana](https://grafana.com/) can be used to make interactive dashboards, checkout [this example showing Point data](https://blog.timescale.com/blog/grafana-variables-101/). Note there is an [AWS managed service for Grafana](https://aws.amazon.com/grafana/)
* [litestream](https://litestream.io/) -> Continuously stream SQLite changes to S3-compatible storage
* [ImageFusion)](https://github.com/JohMast/ImageFusion) -> Temporal fusion of raster image time-Series
* [nvtop](https://github.com/Syllo/nvtop) -> NVIDIA GPUs htop like monitoring tool
* [rgis](https://github.com/frewsxcv/rgis) -> Geospatial data viewer written in Rust
* [aerialbot](https://github.com/doersino/aerialbot) -> A simple yet highly configurable bot that tweets geotagged aerial imagery of a random location in the world
* [SatDump](https://github.com/altillimity/SatDump) -> A generic satellite data processing software.
* [EOdal](https://github.com/EOA-team/eodal) -> a Python library enabling the acquisition, organization, and analysis of EO data in a completely open-source manner within a unified framework.
* [SRAI](https://github.com/kraina-ai/srai) -> Spatial Representations for Artificial Intelligence aims to provide simple and efficient solutions to geospatial problems that are accessible to everybody and reusable in various contexts where geospatial data can be used.
* [rico-hdl](https://github.com/kai-tub/rico-hdl) -> A fast and easy-to-use **r**emote sensing **i**mage format **co**nverter for **h**igh-throughput **d**eep-**l**earning (rico-hdl).


## Image dataset creation
Many datasets on kaggle & elsewhere have been created by screen-clipping Google Maps or browsing web portals. The tools below are to create datasets programatically
* [MapTilesDownloader](https://github.com/AliFlux/MapTilesDownloader) -> A super easy to use map tiles downloader built using Python
* [jimutmap](https://github.com/Jimut123/jimutmap) -> get enormous amount of high resolution satellite images from apple / google maps quickly through multi-threading
* [google-maps-downloader](https://github.com/yildirimcagatay34/google-maps-downloader) -> A short python script that downloads satellite imagery from Google Maps
* [ExtractSatelliteImagesFromCSV](https://github.com/thewati/ExtractSatelliteImagesFromCSV) -> extract satellite images using a CSV file that contains latitude and longitude, uses mapbox
* [sentinelsat](https://github.com/sentinelsat/sentinelsat) -> Search and download Copernicus Sentinel satellite images
* [SentinelDownloader](https://github.com/cordmaur/SentinelDownloader) -> a high level wrapper to the SentinelSat that provides an object oriented interface, asynchronous downloading, quickview & simpler searching methods
* [GEES2Downloader](https://github.com/cordmaur/GEES2Downloader) -> Downloader for GEE S2 bands
* [Sentinel-2 satellite tiles images downloader from Copernicus](https://github.com/flaviostutz/sentinelloader) -> Minimizes data download and combines multiple tiles to return a single area of interest
* [felicette](https://github.com/plant99/felicette) -> Satellite imagery for dummies. Generate JPEG earth imagery from coordinates/location name with publicly available satellite data
* [Easy Landsat Download](https://github.com/dgketchum/Landsat578)
* [A simple python scrapper to get satellite images of Africa, Europe and Oceania's weather using the Sat24 website](https://github.com/luistripa/sat24-image-scrapper)
* [RGISTools](https://github.com/spatialstatisticsupna/RGISTools) -> Tools for Downloading, Customizing, and Processing Time Series of Satellite Images from Landsat, MODIS, and Sentinel
* [DeepSatData](https://github.com/michaeltrs/DeepSatData) -> Automatically create machine learning datasets from satellite images
* [landsat_ingestor](https://github.com/landsat-pds/landsat_ingestor) -> Scripts and other artifacts for landsat data ingestion into Amazon public hosting
* [satpy](https://github.com/pytroll/satpy) -> a python library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats
* [GIBS-Downloader](https://github.com/spaceml-org/GIBS-Downloader) -> a command-line tool which facilitates the downloading of NASA satellite imagery and offers different functionalities in order to prepare the images for training in a machine learning pipeline
* [eodag](https://github.com/CS-SI/eodag) -> Earth Observation Data Access Gateway
* [pylandsat](https://github.com/yannforget/pylandsat) -> Search, download, and preprocess Landsat imagery
* [landsatxplore](https://github.com/yannforget/landsatxplore) -> Search and download Landsat scenes from EarthExplorer
* [OpenSarToolkit](https://github.com/ESA-PhiLab/OpenSarToolkit) -> High-level functionality for the inventory, download and pre-processing of Sentinel-1 data in the python language
* [lsru](https://github.com/loicdtx/lsru) -> Query and Order Landsat Surface Reflectance data via ESPA
* [eoreader](https://github.com/sertit/eoreader) -> Remote-sensing opensource python library reading optical and SAR sensors, loading and stacking bands, clouds, DEM and index in a sensor-agnostic way
* [Export thumbnails from Earth Engine](https://gorelick.medium.com/fast-er-downloads-a2abd512aa26)
* [deepsentinel-osm](https://github.com/Lkruitwagen/deepsentinel-osm) -> A repository to generate land cover labels from OpenStreetMap
* [img2dataset](https://github.com/rom1504/img2dataset) -> Easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap (OSM) Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> a library for creating machine-learning ready data by pairing satellite images with OpenStreetMap (OSM) vector data
* [sentinel2tools](https://github.com/QuantuMobileSoftware/sentinel2tools) -> downloading & basic processing of Sentinel 2 imagesry. Read [Sentinel2tools: simple lib for downloading Sentinel-2 satellite images](https://medium.com/geekculture/sentinel2tools-simple-lib-for-downloading-sentinel-2-satellite-images-f8a6be3ee894)
* [Aerial-Satellite-Imagery-Retrieval](https://github.com/chiragkhandhar/Aerial-Satellite-Imagery-Retrieval) -> A program using Bing maps tile system to automatically download Aerial / Satellite Imagery given a lat/lon bounding box and level of detail
* [google-maps-at-88-mph](https://github.com/doersino/google-maps-at-88-mph) -> Google Maps keeps old satellite imagery around for a while – this tool collects what's available for a user-specified region in the form of a GIF
* [srtmDownloader](https://github.com/Abdi-Ghasem/srtmDownloader) -> Python library (multi-threaded) for retrieving SRTM elevation map of CGIAR-CSI
* [ImageDatasetViz](https://github.com/vfdev-5/ImageDatasetViz) -> create a mosaic of images in a dataset for previewing purposes
* [landsatlinks](https://github.com/ernstste/landsatlinks) -> A simple CLI interface to generate download urls for Landsat Collection 2 Level 1 product bundles
* [pyeo](https://github.com/clcr/pyeo) -> a set of portable, extensible and modular Python scripts for machine learning in earth observation and GIS, including downloading, preprocessing, creation of base layers, classification and validation.
* [metaearth](https://github.com/bair-climate-initiative/metaearth) -> Download and access remote sensing data from any platform
* [geoget](https://github.com/mnpinto/geoget) -> Download geodata for anywhere in Earth via ladsweb.modaps.eosdis.nasa.gov
* [geeml](https://github.com/Geethen/geeml) -> A python package to extract Google Earth Engine data for machine learning
* [xlandsat](https://github.com/compgeolab/xlandsat) -> A Python package for handling Landsat scenes from EarthExplorer with xarray
* [tms2geotiff](https://github.com/gumblex/tms2geotiff) -> Download tiles from Tile Map Server (online maps) and make a large geo-referenced image
* [s2-chips](https://github.com/tharlestsa/s2-chips) -> efficiently extracts satellite imagery chips from Sentinel-2 datasets based on given geo-coordinates from a GeoJSON file. Uses Ray for parallel processing

## Image chipping/tiling & merging
Since raw images can be very large, it is usually necessary to chip/tile them into smaller images before annotation & training
* [image_slicer](https://github.com/samdobson/image_slicer) -> Split images into tiles. Join the tiles back together
* [tiler by nuno-faria](https://github.com/nuno-faria/tiler) -> split images into tiles and merge tiles into a large image
* [tiler by the-lay](https://github.com/the-lay/tiler) -> N-dimensional NumPy array tiling and merging with overlapping, padding and tapering
* [xbatcher](https://github.com/pangeo-data/xbatcher) -> Xbatcher is a small library for iterating xarray DataArrays in batches. The goal is to make it easy to feed xarray datasets to machine learning libraries such as Keras
* [GeoTagged_ImageChip](https://github.com/Hejarshahabi/GeoTagged_ImageChip) -> A simple script to create geo tagged image chips from high resolution RS iamges for training deep learning models such as Unet
* [geotiff-crop-dataset](https://github.com/tayden/geotiff-crop-dataset) -> A Pytorch Dataloader for tif image files that dynamically crops the image
* [Train-Test-Validation-Dataset-Generation](https://github.com/salarghaffarian/Train-Test-Validation-Dataset-Generation) ->  app to crop images and create small patches of a large image e.g. Satellite/Aerial Images, which will then be used for training and testing Deep Learning models specifically semantic segmentation models
* [satproc](https://github.com/dymaxionlabs/satproc) -> Python library and CLI tools for processing geospatial imagery for ML
* [Sliding Window](https://github.com/adamrehn/slidingwindow) ->  break large images into a series of smaller chunks
* [patchify](https://github.com/dovahcrow/patchify.py) -> A library that helps you split image into small, overlappable patches, and merge patches into original image
* [split-rs-data](https://github.com/Youssef-Harby/split-rs-data) -> Divide remote sensing images and their labels into data sets of specified size
* [image-reconstructor-patches](https://github.com/marijavella/image-reconstructor-patches) -> Reconstruct Image from Patches with a Variable Stride
* [rpc_cropper](https://github.com/carlodef/rpc_cropper) -> A small standalone tool to crop satellite images and their RPC
* [geotile](https://github.com/iamtekson/geotile) -> python library for tiling the geographic raster data
* [GeoPatch](https://github.com/Hejarshahabi/GeoPatch) -> generating patches from remote sensing data
* [ImageTilingUtils](https://github.com/vfdev-5/ImageTilingUtils) -> Minimalistic set of image reader agnostic tools to easily iterate over large images
* [split_raster](https://github.com/cuicaihao/split_raster) -> Creates a tiled output from an input raster dataset. pip installable
* [SAHI](https://github.com/obss/sahi) -> Utilties for slicing COCO formatted annotations and image files, performing sliced inference using MMDetection, Detectron2, YOLOv5, HuggingFace detectors and calculating AP over image slices.
* [geo2ml](https://github.com/mayrajeo/geo2ml) -> Python library and module for converting earth observation data to be suitable for machine learning models, Converting vector data to COCO and YOLO formats and creating required dataset files, Rasterizing polygon geometries for semantic segmentation tasks, Tiling larger rasters and shapefiles into smaller patches
* [FlipnSlide](https://github.com/elliesch/flipnslide) -> a concise tiling and augmentation strategy to prepare large scientific images for use with GPU-enabled algorithms. Outputs PyTorch-ready preprocessed datasets from a single large image.

## Image processing, handling, manipulation
* [Pillow is the Python Imaging Library](https://pillow.readthedocs.io/en/stable/) -> this will be your go-to package for image manipulation in python
* [opencv-python](https://github.com/opencv/opencv-python) is pre-built CPU-only OpenCV packages for Python
* [kornia](https://github.com/kornia/kornia) is a differentiable computer vision library for PyTorch, like openCV but on the GPU. Perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.
* [tifffile](https://github.com/cgohlke/tifffile) -> Read and write TIFF files
* [xtiff](https://github.com/BodenmillerGroup/xtiff) -> A small Python 3 library for writing multi-channel TIFF stacks
* [geotiff](https://github.com/Open-Source-Agriculture/geotiff) -> A noGDAL tool for reading and writing geotiff files
* [geolabel-maker](https://github.com/makinacorpus/geolabel-maker) -> combine satellite or aerial imagery with vector spatial data to create your own ground-truth dataset in the COCO format for deep-learning models
* [imagehash](https://github.com/JohannesBuchner/imagehash) -> Image hashes tell whether two images look nearly identical
* [fake-geo-images](https://github.com/up42/fake-geo-images) -> A module to programmatically create geotiff images which can be used for unit tests
* [imagededup](https://github.com/idealo/imagededup) -> Finding duplicate images made easy! Uses perceptual hashing
* [duplicate-img-detection](https://github.com/mattpodolak/duplicate-img-detection) -> A basic duplicate image detection service using perceptual image hash functions and nearest neighbor search, implemented using faiss, fastapi, and imagehash
* [rmstripes](https://github.com/DHI-GRAS/rmstripes) -> Remove stripes from images with a combined wavelet/FFT approach
* [activeloopai Hub](https://github.com/activeloopai/hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Scalable data pipelines.
* [sewar](https://github.com/andrewekhalel/sewar) -> All image quality metrics you need in one package
* [Satellite imagery label tool](https://github.com/calebrob6/labeling-tool) -> provides an easy way to collect a random sample of labels over a given scene of satellite imagery
* [Missing-Pixel-Filler](https://github.com/spaceml-org/Missing-Pixel-Filler) -> given images that may contain missing data regions (like satellite imagery with swath gaps), returns these images with the regions filled
* [color_range_filter](https://github.com/developmentseed/color_range_filter) -> a script that allows us to find range of colors in images using openCV, and then convert them into geo vectors
* [eo4ai](https://github.com/ESA-PhiLab/eo4ai) -> easy-to-use tools for preprocessing datasets for image segmentation tasks in Earth Observation
* [rasterix](https://github.com/mogasw/rasterix) -> a cross-platform utility built around the GDAL library and the Qt framework designed to process geospatial raster data
* [datumaro](https://github.com/openvinotoolkit/datumaro) -> Dataset Management Framework, a Python library and a CLI tool to build, analyze and manage Computer Vision datasets
* [sentinelPot](https://github.com/LLeiSong/sentinelPot) -> a python package to preprocess Sentinel 1&2 imagery
* [ImageAnalysis](https://github.com/UASLab/ImageAnalysis) -> Aerial imagery analysis, processing, and presentation scripts.
* [rastertodataframe](https://github.com/mblackgeo/rastertodataframe) -> Convert any GDAL compatible raster to a Pandas DataFrame
* [yeoda](https://github.com/TUW-GEO/yeoda) -> provides lower and higher-level data cube classes to work with well-defined and structured earth observation data
* [tiles-to-tiff](https://github.com/jimutt/tiles-to-tiff) -> Python script for converting XYZ raster tiles for slippy maps to a georeferenced TIFF image
* [telluric](https://github.com/satellogic/telluric) -> a Python library to manage vector and raster geospatial data in an interactive and easy way
* [Sniffer](https://github.com/2320sharon/Sniffer) -> A python application for sorting through geospatial imagery
* [pyjeo](https://github.com/ec-jrc/jeolib-pyjeo) -> a library for image processing for geospatial data implemented in JRC Ispra, with [paper](https://www.mdpi.com/2220-9964/8/10/461)
* [vpv](https://github.com/kidanger/vpv) -> Image viewer designed for image processing experts
* [arop](https://github.com/george-silva/arop) -> Automated Registration and Orthorectification Package
* [satellite_image](https://github.com/dgketchum/satellite_image) -> Python package to process images from Landsat satellites and return geographic information, cloud mask, numpy array, geotiff
* [large_image](https://github.com/girder/large_image) -> Python modules to work with large multiresolution images
* [ResizeRight](https://github.com/assafshocher/ResizeRight) -> The correct way to resize images or tensors. For Numpy or Pytorch (differentiable)
* [pysat](https://github.com/pysat/pysat) -> a package providing a simple and flexible interface for downloading, loading, cleaning, managing, processing, and analyzing scientific measurements
* [plcompositor](https://github.com/planetlabs/plcompositor) -> c++ tool from Planet to create seamless and cloudless image mosaics from deep stacks of satellite imagery

## Image augmentation packages
Image augmentation is a technique used to expand a training dataset in order to improve ability of the model to generalise
* [AugLy](https://github.com/facebookresearch/AugLy) -> A data augmentations library for audio, image, text, and video. By Facebook
* [albumentations](https://github.com/albumentations-team/albumentations) -> Fast image augmentation library and an easy-to-use wrapper around other libraries
* [FoHIS](https://github.com/noahzn/FoHIS) -> Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
* [Kornia](https://kornia.readthedocs.io/en/latest/augmentation.html) provides augmentation on the GPU
* [toolbox by ming71](https://github.com/ming71/toolbox) -> various cv tools, such as label tools, data augmentation, label conversion, etc.
* [AstroAugmentations](https://github.com/mb010/AstroAugmentations) -> augmentations designed around astronomical instruments
* [Chessmix](https://github.com/matheusbarrosp/chessmix) -> data augmentation method for remote sensing semantic segmentation
* [satellite_object_augmentation](https://github.com/LanaLana/satellite_object_augmentation) -> Object-based augmentation for remote sensing images segmentation via CNN
* [hypernet](https://github.com/ESA-PhiLab/hypernet) -> hyperspectral data augmentation

## Image formats, data management and catalogues
* [GeoServer](http://geoserver.org/) -> an open source server for sharing geospatial data
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* https://terria.io/ for pretty catalogues
* Large datasets may come in HDF5 format, can view with -> https://www.hdfgroup.org/downloads/hdfview/
* Climate data is often in netcdf format, which can be opened using xarray
* [TileDB](https://tiledb.com/) -> a 'Universal Data Engine' to store, analyze and share any data (beyond tables), with any API or tool (beyond SQL) at planet-scale (beyond clusters), open source and managed options.
* Read about [Serverless PostGIS on AWS Aurora](https://blog.addresscloud.com/serverless-postgis/)
* [Hub](https://github.com/activeloopai/Hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Read [Faster Machine Learning Using Hub by Activeloop: A code walkthrough of using the hub package for satellite imagery](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005)
* [A Comparison of Spatial Functions: PostGIS, Athena, PrestoDB, BigQuery vs RedShift](https://ual.sg/post/2020/07/03/a-comparison-of-spatial-functions-postgis-athena-prestodb-bigquery-vs-redshift/)
* [Unfolded Studio](https://studio.unfolded.ai/) -> visualization platform building on open source geospatial technologies including kepler.gl, deck.gl and H3. Processing is performed browser side enabling very responsive visualisations.
* [DroneDB](https://github.com/DroneDB/DroneDB) -> can index and extract useful information from the EXIF/XMP tags of aerial images to display things like image footprint, flight path and image GPS location
* [embeddinghub](https://github.com/featureform/embeddinghub) -> A vector database for machine learning embeddings
* [Resonant GeoData](https://github.com/ResonantGeoData/ResonantGeoData/) -> a Django application well suited for catalogging and searching annotated geospatial imagery, shapefiles, and full motion video datasets
* [fastdup](https://github.com/visualdatabase/fastdup) -> a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images
* [Nucleus](https://dashboard.scale.com/nucleus/) is a platform for image dataset management with advanced features including [autotagging](https://nucleus.scale.com/docs/introduction-to-autotag) and finding [instances with mismatched predictions & annotations](https://nucleus.scale.com/docs/find-inaccurate-predictions)

## General utilities
Scripts and command line applications
* [geospatial-cli](https://github.com/JakobMiksch/geospatial-cli) -> a collection of geospatial programs with commandline interface
* [PyShp](https://github.com/GeospatialPython/pyshp) -> The Python Shapefile Library (PyShp) reads and writes Shapefiles in pure Python
* [s2p](https://github.com/cmla/s2p) -> a Python library and command line tool that implements a stereo pipeline which produces elevation models from images taken by high resolution optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos
* [EarthPy](https://github.com/earthlab/earthpy) -> A set of helper functions to make working with spatial data in open source tools easier. read[Exploratory Data Analysis (EDA) on Satellite Imagery Using EarthPy](https://towardsdatascience.com/exploratory-data-analysis-eda-on-satellite-imagery-using-earthpy-c0e186fe4293)
* [pygeometa](https://geopython.github.io/pygeometa/) -> provides a lightweight and Pythonic approach for users to easily create geospatial metadata in standards-based formats using simple configuration files
* [GEOS](https://geos.readthedocs.io/en/latest/index.html) -> Google Earth Overlay Server (GEOS) is a python-based server for creating Google Earth overlays of tiled maps. Your can also display maps in the web browser, measure distances and print maps as high-quality PDF’s.
* [GeoDjango](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/) intends to be a world-class geographic Web framework. Its goal is to make it as easy as possible to build GIS Web applications and harness the power of spatially enabled data. [Some features of GDAL are supported.](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/gdal/)
* [rasterstats](https://pythonhosted.org/rasterstats/) -> summarize geospatial raster datasets based on vector geometries
* [turfpy](https://turfpy.readthedocs.io/en/latest/index.html) -> a Python library for performing geospatial data analysis which reimplements turf.js
* [rsgislib](https://github.com/remotesensinginfo/rsgislib) -> Remote Sensing and GIS Software Library; python module tools for processing spatial and image data
* [eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html) -> seamlessly access and process spatio-temporal image sequences acquired by any satellite fleet in a timely and automatic manner. See [eo-learn-examples](https://github.com/sentinel-hub/eo-learn-examples)
* [RStoolbox: Tools for Remote Sensing Data Analysis in R](https://bleutner.github.io/RStoolbox/)
* [nd](https://github.com/jnhansen/nd) -> Framework for the analysis of n-dimensional, multivariate Earth Observation data, built on xarray
* [reverse-geocoder](https://github.com/thampiman/reverse-geocoder) -> a fast, offline reverse geocoder in Python
* [MuseoToolBox](https://github.com/nkarasiak/MuseoToolBox) -> a python library to simplify the use of raster/vector, especially for machine learning and remote sensing
* [py6s](https://py6s.readthedocs.io/en/latest/) -> an interface to the Second Simulation of the Satellite Signal in the Solar Spectrum (6S) atmospheric Radiative Transfer Model
* [timvt](https://github.com/developmentseed/timvt) -> PostGIS based Vector Tile server built on top of the modern and fast FastAPI framework
* [titiler](https://github.com/developmentseed/titiler) -> A dynamic Web Map tile server using FastAPI
* [BRAILS](https://github.com/NHERI-SimCenter/BRAILS) -> an AI-based pipeline for city-scale building information modelling (BIM)
* [color-thief-py](https://github.com/fengsp/color-thief-py) -> Grabs the dominant color or a representative color palette from an image
* [force](https://github.com/davidfrantz/force) -> an all-in-one processing engine for medium-resolution Earth Observation image archives
* [mapwarper](https://github.com/timwaters/mapwarper) -> an open source map geo-rectification, warping and georeferencing application
* [sarpy](https://github.com/ngageoint/sarpy) -> A basic Python library to demonstrate reading, writing, display, and simple processing of complex SAR data using the NGA SICD standard
* [buzzard](https://github.com/earthcube-lab/buzzard) -> Advanced raster and geometry manipulations
* [sentinel1denoised](https://github.com/nansencenter/sentinel1denoised) -> Thermal noise subtraction, scalloping correction, angular correction
* [RStoolbox](https://github.com/bleutner/RStoolbox) -> Remote Sensing Data Analysis in R
* [kart](https://github.com/koordinates/kart) -> Distributed version-control for geospatial and tabular data
* [picogeojson](https://github.com/fortyninemaps/picogeojson) -> a Python library for reading, writing, and working with GeoJSON
* [shareloc](https://github.com/CNES/shareloc) -> a simple remote sensing geometric library, to perform image coordinates projections between sensor and ground and vice versa
* [geoblaze](https://github.com/GeoTIFF/geoblaze) -> Blazing Fast JavaScript Raster Processing Engine
* [nasa-wildfires](https://github.com/datadesk/nasa-wildfires) -> Download wildfire hotspots detected by NASA satellites and the Fire Information for Resource Management System (FIRMS)
* [SSGP-toolbox](https://github.com/Dreamlone/SSGP-toolbox) -> Simple Spatial Gapfilling Processor. Toolbox for filling gaps in spatial datasets
* [imgreg2D](https://github.com/BrancoLab/imgreg2D) -> 2D image registration in python, using napari
* [georust](https://github.com/georust) -> A collection of geospatial tools and libraries written in Rust
* [DataPillager](https://github.com/gdherbert/DataPillager) -> Download data from Esri REST service
* [litexplore](https://github.com/litements/litexplore) -> a Python web app that lets you explore remote SQLite databases over SSH connections
* [tifeatures](https://github.com/developmentseed/tifeatures) -> Simple and Fast Geospatial Features API for PostGIS
* [pyroSAR](https://github.com/johntruckenbrodt/pyroSAR) -> framework for large-scale SAR satellite data processing
* [S1_NRB](https://github.com/SAR-ARD/S1_NRB) -> A prototype processor for the Sentinel-1 Normalised Radar Backscatter product
* [AGBench](https://github.com/gyrrei/AGBench) -> a Python library that benchmarks satellite-based aboveground biomass or carbon estimate maps
* [mbtiles-s3-server](https://github.com/uktrade/mbtiles-s3-server) -> Python server to on-the-fly extract and serve vector tiles from an mbtiles file on S3
* [matico](https://github.com/Matico-Platform/matico) -> a set of tools and services that allow users to manage geospatial datasets, build APIs that use those datasets and full geospatial applications with little to no code
* [gmtsar](https://github.com/mobigroup/gmtsar) -> easy and fast satellite interferometry (InSAR) processing
* [image_tiles](https://github.com/moonshinelabs-ai/image_tiles) -> a simple but flexible tool to view a folder full of images on your web browser
* [sen2mosaic](https://github.com/smfm-project/sen2mosaic) -> a set of tools to aid in the production of large-scale cloud-free seasonal mosaic products from Sentinel-2 data

## Low level numerical & data formats
* [xarray](http://xarray.pydata.org/en/stable/) -> N-D labeled arrays and datasets. Read [Handling multi-temporal satellite images with Xarray](https://medium.com/@bonnefond.virginie/handling-multi-temporal-satellite-images-with-xarray-30d142d3391). Checkout [xarray_leaflet](https://github.com/davidbrochart/xarray_leaflet) for tiled map plotting and [sklearn-xarray](https://github.com/phausamann/sklearn-xarray) for metadata-aware machine learning. Publish Xarray Datasets via a REST API uisng [xpublish](https://github.com/xarray-contrib/xpublish)
* [wxee](https://github.com/aazuspan/wxee) -> Export data from GEE to xarray using wxee then train with pytorch or tensorflow models. Useful since GEE only suports tfrecord export natively
* [xarray-spatial](https://github.com/makepath/xarray-spatial) -> Fast, Accurate Python library for Raster Operations. Implements algorithms using Numba and Dask, free of GDAL
* [datatree](https://github.com/xarray-contrib/datatree) -> WIP implementation of a tree-like hierarchical data structure for xarray
* [xarray-beam](https://github.com/google/xarray-beam) -> Distributed Xarray with Apache Beam by Google
* [Geowombat](https://geowombat.readthedocs.io/) -> geo-utilities applied to air and space-borne imagery, uses Rasterio, Xarray and Dask for I/O and distributed computing with named coordinates. [Create Land Use Classification using Geowombat & Sklearn](https://pygis.io/docs/f_rs_ml_predict.html)
* [NumpyTiles](https://github.com/planetlabs/numpytiles-spec) -> a specification for providing multiband full-bit depth raster data in the browser
* [Zarr](https://zarr.readthedocs.io/en/stable/) -> Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. Zarr depends on NumPy
* [geoparquet](https://github.com/opengeospatial/geoparquet) -> Specification for storing geospatial vector data (point, line, polygon) in Parquet
* [TFRecord reader for PyTorch](https://github.com/vahidk/tfrecord)

## Graphing and visualisation
* [hvplot](https://hvplot.holoviz.org/) -> A high-level plotting API for the PyData ecosystem built on HoloViews. Allows overlaying data on map tiles, see [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727)
* [Pyviz](https://examples.pyviz.org/) examples include several interesting geospatial visualisations
* [napari](https://napari.org) -> napari is a fast, interactive, multi-dimensional image viewer for Python. It’s designed for browsing, annotating, and analyzing large multi-dimensional images. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools. Note that to view a 3GB COG I had to install the [napari-tifffile-reader](https://github.com/GenevieveBuckley/napari-tifffile-reader) plugin.
* [pixel-adjust](https://github.com/cisaacstern/pixel-adjust) -> Interactively select and adjust specific pixels or regions within a single-band raster. Built with rasterio, matplotlib, and panel.
* [Plotly Dash](https://plotly.com/dash/) can be used for making interactive dashboards
* [folium](https://python-visualization.github.io/folium/) -> a python wrapper to the excellent [leaflet.js](https://leafletjs.com/) which makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. Also checkout the [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) component for adding folium maps to your streamlit apps
* [ipyearth](https://github.com/davidbrochart/ipyearth) -> An IPython Widget for Earth Maps
* [geopandas-view](https://github.com/martinfleis/geopandas-view) -> Interactive exploration of GeoPandas GeoDataFrames
* [geogif](https://github.com/gjoseph92/geogif) -> Turn xarray timestacks into GIFs
* [leafmap](https://github.com/giswqs/leafmap) -> geospatial analysis and interactive mapping with minimal coding in a Jupyter environment
* [xmovie](https://github.com/jbusecke/xmovie) -> A simple way of creating movies from xarray objects
* [acquisition-time](https://github.com/charlotte-pel/acquisition-time) -> Drawing (Satellite) acquisition dates in a timeline
* [splot](https://github.com/pysal/splot) -> Lightweight plotting for geospatial analysis in PySAL
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Tools to Design or Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)
* [AstronomicAL](https://github.com/grant-m-s/AstronomicAL) -> An interactive dashboard for visualisation, integration and classification of data using Active Learning
* [pyodi](https://github.com/Gradiant/pyodi) -> A simple tool for explore your object detection dataset
* [Interactive-TSNE](https://github.com/spaceml-org/Interactive-TSNE) -> a tool that provides a way to visually view a PyTorch model's feature representation for better embedding space interpretability
* [fastgradio](https://github.com/aliabd/fastgradio) -> Build fast gradio demos of fastai learners
* [pysheds](https://github.com/mdbartos/pysheds) -> Simple and fast watershed delineation in python
* [mapboxgl-jupyter](https://github.com/mapbox/mapboxgl-jupyter) -> Use Mapbox GL JS to visualize data in a Python Jupyter notebook
* [cartoframes](https://github.com/CartoDB/cartoframes) -> integrate CARTO maps, analysis, and data services into data science workflows
* [datashader](https://datashader.org/) -> create meaningful representations of large datasets quickly and flexibly. Read [Creating Visual Narratives from Geospatial Data Using Open-Source Technology Maxar blog post](https://blog.maxar.com/tech-and-tradecraft/2021/creating-visual-narratives-from-geospatial-data-using-open-source-technology)
* [Kaleido](https://github.com/plotly/Kaleido) -> Fast static image export for web-based visualization libraries with zero dependencies
* [Embedding Projector in Wandb](https://docs.wandb.ai/ref/app/features/panels/weave/embedding-projector) -> allows users to plot multi-dimensional embeddings on a 2D plane using common dimension reduction algorithms like PCA, UMAP, and t-SNE
* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) -> Latex code for making neural networks diagrams
* [Damage Assessment Visualizer](https://github.com/microsoft/Nonprofits/tree/master/Damage%20Assessment%20Visualizer) -> leverages satellite imagery from a disaster region to visualize conditions of building and structures before and after a disaster
* [NN-SVG](https://github.com/alexlenail/NN-SVG) -> is a tool for creating Neural Network (NN) architecture drawings parametrically rather than manually
* [bbox-visualizer](https://github.com/shoumikchow/bbox-visualizer) -> Make drawing and labeling bounding boxes easy as cake
* [jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget) -> A Jupyter widget for annotating images with bounding boxes
* [EOmaps](https://github.com/raphaelquast/EOmaps) -> A library to create interactive maps of geographical datasets
* [H3-Pandas](https://github.com/DahnJ/H3-Pandas) -> Integrates H3 with GeoPandas and Pandas
* [gmplot](https://github.com/gmplot/gmplot) -> a matplotlib-like interface to render all the data you'd like on top of Google Maps
* [NPYViewer](https://github.com/csmailis/NPYViewer) ->  a simple GUI tool that provides multiple ways to view `.npy` files containing 2D NumPy Arrays
* [pyGEOVis](https://github.com/geoyee/pyGEOVis) -> Visualize geo-tiff/json based on folium
* [bokeh-tiler](https://github.com/avanetten/bokeh-tiler) -> Tile large geospatial images for use in Bokeh. Read [Serving up SpaceNet Imagery for Bokeh](https://medium.com/geodesic/serving-up-spacenet-imagery-for-bokeh-e85b8fffe05)
* [torchshow](https://github.com/xwying/torchshow) -> Visualize PyTorch tensor in one-line of code
* [pixels](https://github.com/jwasilgeo/pixels) -> Mapping and charting pixels from remote sensing Earth observation data with JavaScript
* [MulimgViewer](https://github.com/nachifur/MulimgViewer) -> a multi-image viewer that can open multiple images in one interface
* [cnn-explainer](https://github.com/poloclub/cnn-explainer) -> Learning Convolutional Neural Networks with Interactive Visualization
* [Overlay-GeoTiff-Raster-with-nodata-On-Interactive-Map](https://github.com/royalosyin/Overlay-GeoTiff-Raster-with-nodata-On-Interactive-Map)
* [shapefile2gif](https://github.com/johannesuhl/shapefile2gif) -> Given a shapefile with time-annotated vector objects (e.g., building footprints + construction year), this script will automatically create an animated GIF illustrating the dynamics for a user-specified period of time
* [insat3d_imagen](https://github.com/rupeshs/insat3d_imagen) -> Processes INSAT HDF file and generates satellite images
* [pygieons](https://github.com/pygieons/pygieons) -> A simple package to visualize and keep track of GIS and Earth Observation libraries in Python
* [regionmask](https://github.com/regionmask/regionmask) -> Create masks of geographical regions for arbitrary longitude and latitude grids
* [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
* [terrain-diffusion-app](https://github.com/sshh12/terrain-diffusion-app) -> An infinite collaborative inpainter which allows users to dynamic generate satellite-realistic map tiles

## Algorithms
* [WaterDetect](https://github.com/cordmaur/WaterDetect) -> an end-to-end algorithm to generate open water cover mask, specially conceived for L2A Sentinel 2 imagery. It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.
* [GatorSense Hyperspectral Image Analysis Toolkit](https://github.com/GatorSense/hsi_toolkit_py) -> This repo contains algorithms for Anomaly Detectors, Classifiers, Dimensionality Reduction, Endmember Extraction, Signature Detectors, Spectral Indices
* [detectree](https://github.com/martibosch/detectree) -> Tree detection from aerial imagery
* [pylandstats](https://github.com/martibosch/pylandstats) -> compute landscape metrics
* [dg-calibration](https://github.com/DHI-GRAS/dg-calibration) -> Coefficients and functions for calibrating DigitalGlobe imagery
* [python-fmask](https://github.com/ubarsc/python-fmask) -> Implementation in Python of the cloud and shadow algorithms known collectively as Fmask
* [pyshepseg](https://github.com/ubarsc/pyshepseg) -> Python implementation of image segmentation algorithm of Shepherd et al (2019) Operational Large-Scale Segmentation of Imagery Based on Iterative Elimination.
* [Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images](https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images) -> shadow detection and correction algorithm
* [faiss](https://github.com/facebookresearch/faiss) -> A library for efficient similarity search and clustering of dense vectors, e.g. image embeddings
* [awesome-spectral-indices](https://github.com/davemlz/awesome-spectral-indices) -> A ready-to-use curated list of Spectral Indices for Remote Sensing applications
* [urban-footprinter](https://github.com/martibosch/urban-footprinter) -> A convolution-based approach to detect urban extents from raster datasets
* [ocean_color](https://github.com/marrs-lab/ocean_color) -> Tools and algorithms for drone and satellite based ocean color science
* [poliastro](https://github.com/poliastro/poliastro) -> pure Python library for interactive Astrodynamics and Orbital Mechanics, with a focus on ease of use, speed, and quick visualization
* [acolite](https://github.com/acolite/acolite) -> generic atmospheric correction module
* [pmapper](https://github.com/nasa-jpl/pmapper) -> a super-resolution and deconvolution toolkit for python. PMAP stands for Poisson Maximum A-Posteriori, a highly flexible and adaptable algorithm for these problems
* [pylandtemp](https://github.com/pylandtemp/pylandtemp) -> Algorithms for computing global land surface temperature and emissivity from NASA's Landsat satellite images with Python
* [sarsen](https://github.com/bopen/sarsen) -> Algorithms and utilities for Synthetic Aperture Radar (SAR) sensors
* [sun-position](https://github.com/s-bear/sun-position) -> code for computing sun position
* [simple_ortho](https://github.com/dugalh/simple_ortho) -> Fast and simple orthorectification of images with known DEM and camera model
* [imageResolution](https://github.com/geojames/imageResolution) -> Simple spatial resolution calculator for nadir & oblique aerial imagery
* [Spectral-Clustering](https://github.com/zhangyk8/Spectral-Clustering) -> normalized and unnormalized spectral clustering algorithms
* [Fogpy](https://github.com/pytroll/fogpy) -> nowcasting of fog and low stratus clouds
* [orthorectification](https://github.com/mpfaffenberger/orthorectification) -> Orthorectification in Python. Note that all of this functionality already exists in libraries like GDAL and others. The goal of this codebase was to present and deep dive into these subroutines
* [Flood-Severity-Estimation](https://github.com/jorgemspereira/Flood-Severity-Estimation) -> estimate the height of the water in geo-referenced photos that depict floods using DEMs from JAXA
* [coastline-extraction](https://github.com/Ricardo-C-Oliveira/coastline-extraction) -> Methods to identify and extract coastline from remote sensed data
* [Near real-time shadow detection and removal in remote sensing imagery application](https://github.com/BIT-zhwang/remote-sensing-image-shadow-detection-and-removal)
* [image-registration](https://github.com/satish1901/image-registration) -> using Point Feature Detection, Normalized DLT, RANSAC & Image Warping
* [pyTSEB](https://github.com/hectornieto/pyTSEB) -> A python Two Source Energy Balance model for estimation of evapotranspiration with remote sensing data
* [libpredict](https://github.com/la1k/libpredict) -> satellite orbit prediction library
* [GOTCHA](https://github.com/jveitchmichaelis/gotcha) -> Command line implementation of the GOTCHA stereo matching algorithm
* [SREM](https://github.com/oyam/srem) -> A Simplified and Robust Surface Reflectance Estimation Method for Satellite Imagery
* [kaizen](https://github.com/fuzailpalnak/kaizen) -> A library to map match and help tackle the problem of overlapping/intersecting road and building footprint that arises in the process of map making
* [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope) -> Batch shoreline extraction toolkit for PlanetScope Dove satellite imagery
* [mappymatch](https://github.com/NREL/mappymatch) -> Pure-python package for map matching

## GDAL & Rasterio
So improtant this pair gets their own section. GDAL is THE command line tool for reading and writing raster and vector geospatial data formats. If you are using python you will probably want to use Rasterio which provides a pythonic wrapper for GDAL
* [GDAL](https://gdal.org)
* GDAL is a dependency of Rasterio and can be difficult to build and install. I recommend using conda, brew (on OSX) or docker in these situations
* GDAL docker quickstart: `docker pull osgeo/gdal` then `docker run --rm -v $(pwd):/data/ osgeo/gdal gdalinfo /data/cog.tiff`
* [Even Rouault](https://github.com/rouault) maintains GDAL, please consider [sponsoring him](https://github.com/sponsors/rouault)
* [Rasterio](https://rasterio.readthedocs.io/en/latest/) -> reads and writes GeoTIFF and other raster formats and provides a Python API based on Numpy N-dimensional arrays and GeoJSON. There are a variety of plugins that extend Rasterio functionality.
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [rioxarray](https://github.com/corteva/rioxarray) -> geospatial xarray extension powered by rasterio
* [aws-lambda-docker-rasterio](https://github.com/addresscloud/aws-lambda-docker-rasterio) -> AWS Lambda Container Image with Python Rasterio for querying Cloud Optimised GeoTiffs. See [this presentation](https://blog.addresscloud.com/rasters-revealed-2021/)
* [godal](https://github.com/airbusgeo/godal) -> golang wrapper for GDAL
* [Write rasterio to xarray](https://github.com/robintw/XArrayAndRasterio/blob/master/rasterio_to_xarray.py)
* [Loam: A Client-Side GDAL Wrapper for Javascript](https://github.com/azavea/loam)
* [Short list of useful GDAL commands](https://github.com/MaxLenormand/Data-Science-for-Remote-Sensing) while working in data science for remote sensing
* [gdal-segment](https://github.com/cbalint13/gdal-segment) -> implements various segmentation algorithms over raster images
* [aws-gdal-robot](https://github.com/mblackgeo/aws-gdal-robot) -> A proof of concept implementation of running GDAL based jobs using AWS S3/Lambda/Batch
* [gdal2tiles](https://github.com/tehamalab/gdal2tiles) -> A python library for generating map tiles based on gdal2tiles.py from GDAL project
* [gdal3.js](https://github.com/bugra9/gdal3.js) -> Convert raster and vector geospatial data to various formats and coordinate systems entirely in the browser

## Cloud Optimised GeoTiff (COG)
A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF that supports HTTP range requests, enabling downloading of specific tiles rather than the full file. COG generally work normally in GIS software such as QGIS, but are larger than regular GeoTIFFs
* https://www.cogeo.org/
* [cog-best-practices](https://github.com/pangeo-data/cog-best-practices)
* [COGs in production](https://sean-rennie.medium.com/cogs-in-production-e9a42c7f54e4)
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [aiocogeo](https://github.com/geospatial-jeff/aiocogeo) -> Asynchronous cogeotiff reader (python asyncio)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article Cloud Native Geoprocessing of Earth Observation Satellite Data with Pangeo](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Working with COGS and STAC in python using geemap](https://geemap.org/notebooks/44_cog_stac/)
* [Load, Experiment, and Download Cloud Optimized Geotiffs (COG) using Python with Google Colab](https://towardsdatascience.com/access-satellite-imagery-with-aws-and-google-colab-4660178444f5) -> short read which covers finding COGS, opening with Rasterio and doing some basic manipulations, all in a Colab Notebook.
* [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727) -> local COG from public AWS bucket, open with rioxarray, visualise with [hvplot](https://hvplot.holoviz.org/). See [the Jupyter notebook](https://nbviewer.jupyter.org/gist/rsignell-usgs/9657896371bb4f38437505146555264c)
* [aws-lambda-docker-rasterio](https://github.com/addresscloud/aws-lambda-docker-rasterio) -> AWS Lambda Container Image with Python Rasterio for querying Cloud Optimised GeoTiffs. See [this presentation](https://blog.addresscloud.com/rasters-revealed-2021/)
* [cogbeam](https://github.com/GoogleCloudPlatform/cogbeam) -> a python based Apache Beam pipeline, optimized for Google Cloud Dataflow, which aims to expedite the conversion of traditional GeoTIFFs into COGs
* [cogserver](https://github.com/rouault/cogserver) -> Expose a GDAL file as a HTTP accessible on-the-fly COG
* [Displaying a gridded dataset on a web-based map - Step by step guide for displaying large GeoTIFFs, using Holoviews, Bokeh, and Datashader](https://towardsdatascience.com/displaying-a-gridded-dataset-on-a-web-based-map-ad6bbe90247f)
* [cog_worker](https://github.com/Vizzuality/cog_worker) -> Scalable arbitrary analysis on COGs

## SpatioTemporal Asset Catalog specification (STAC)
The STAC specification provides a common metadata specification, API, and catalog format to describe geospatial assets, so they can more easily indexed and discovered.
* Spec at https://github.com/radiantearth/stac-spec
* [STAC 1.0.0: The State of the STAC Software Ecosystem](https://medium.com/radiant-earth-insights/stac-1-0-0-software-ecosystem-updates-da4e800a4973)
* [Planet Disaster Data catalogue](https://planet.stac.cloud/) has the [catalogue source on Github](https://github.com/cholmes/pdd-stac) and uses the [stac-browser](https://github.com/radiantearth/stac-browser)
* [Getting Started with STAC APIs](https://www.azavea.com/blog/2021/04/05/getting-started-with-stac-apis/) intro article
* [SpatioTemporal Asset Catalog API specification](https://github.com/radiantearth/stac-api-spec) -> an API to make geospatial assets openly searchable and crawlable
* [stacindex](https://stacindex.org/) -> STAC Catalogs, Collections, APIs, Software and Tools
* Several useful repos on https://github.com/sat-utils
* [Intake-STAC](https://github.com/intake/intake-stac) -> Intake-STAC provides an opinionated way for users to load Assets from STAC catalogs into the scientific Python ecosystem. It uses the intake-xarray plugin and supports several file formats including GeoTIFF, netCDF, GRIB, and OpenDAP.
* [sat-utils/sat-search](https://github.com/sat-utils/sat-search) -> Sat-search is a Python 3 library and a command line tool for discovering and downloading publicly available satellite imagery using STAC compliant API
* [franklin](https://github.com/azavea/franklin) -> A STAC/OGC API Features Web Service focused on ease-of-use for end-users.
* [stacframes](https://github.com/azavea/stacframes) -> A Python library for working with STAC Catalogs via Pandas DataFrames
* [sat-api-pg](https://github.com/developmentseed/sat-api-pg) -> A Postgres backed STAC API
* [stactools](https://github.com/stac-utils/stactools) -> Command line utility and Python library for STAC
* [pystac](https://github.com/stac-utils/pystac) -> Python library for working with any STAC Catalog
* [STAC Examples for Nightlights data](https://github.com/developmentseed/nightlights_stac_examples) -> minimal example STAC implementation for the [Light Every Night](https://registry.opendata.aws/wb-light-every-night/) dataset of all VIIRS DNB and DMSP-OLS nighttime satellite data
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [stac-fastapi](https://github.com/stac-utils/stac-fastapi) -> STAC API implementation with FastAPI
* [stac-fastapi-elasticsearch](https://github.com/stac-utils/stac-fastapi-elasticsearch) -> Elasticsearch backend for stac-fastapi
* [ml-aoi](https://github.com/stac-extensions/ml-aoi) -> An Item and Collection extension to provide labeled training data for machine learning models
* Discoverable and Reusable ML Workflows for Earth Observation -> [part 1](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-1-e198507b5eaa) and [part 2](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-2-ebe2b4812d5a) with the Geospatial Machine Learning Model Catalog (GMLMC)
* [eoAPI](https://github.com/developmentseed/eoAPI) -> Earth Observation API with STAC + dynamic Raster/Vector Tiler
* [stac-nb](https://github.com/darrenwiens/stac-nb) -> STAC in Jupyter Notebooks
* [xstac](https://github.com/TomAugspurger/xstac) -> Generate STAC Collections from xarray datasets
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [cirrus-geo](https://github.com/cirrus-geo/cirrus-geo) -> a STAC-based processing pipeline
* [stac-interactive-search](https://github.com/calebrob6/stac-interactive-search) -> A simple (browser based) UI for searching STAC APIs
* [easystac](https://github.com/cloudsen12/easystac) -> A Python package for simple STAC queries
* [stacmap](https://github.com/aazuspan/stacmap) -> Explore STAC items with an interactive map
* [odc-stac](https://github.com/opendatacube/odc-stac) -> Load STAC items into xarray Datasets. Process locally or distribute data loading and computation with Dask.
* [AWS Lambda SenCloud Monitoring](https://github.com/ahuarte47/aws-sencloud-monitoring) -> keep up-to-date your own derived data from the Sentinel-2 COG imagery archive using AWS lambda
* [stac-geoparquet](https://github.com/TomAugspurger/stac-geoparquet) -> Convert STAC items to geoparquet
* [labs-gpt-stac](https://github.com/developmentseed/labs-gpt-stac) -> connect ChatGPT to a STAC API backend
* [stac_ipyleaflet](https://github.com/MAAP-Project/stac_ipyleaflet) -> stac_ipyleaflet is a customized version of ipyleaflet built to be an in-jupyter-notebook interactive mapping library that prioritizes access to STAC catalog data
* [prefect-planetary-computer](https://github.com/giorgiobasile/prefect-planetary-computer) -> Prefect integrations with Microsoft Planetary Computer

## OpenStreetMap
[OpenStreetMap](https://www.openstreetmap.org/) (OSM) is a map of the world, created by people like you and free to use under an open license. Quite a few publications use OSM data for annotations & ground truth. Note that the data is created by volunteers and the quality can be variable
* [osmnx](https://github.com/gboeing/osmnx) -> Retrieve, model, analyze, and visualize data from OpenStreetMap
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> downloads OpenStreetMap QA Tile information and satellite imagery tiles and saves them as an `.npz` file for use in machine learning training. This should be used instead of the deprecated [skynet-data](https://github.com/developmentseed/skynet-data)
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”
* [Import OpenStreetMap data into Unreal Engine 4](https://github.com/ue4plugins/StreetMap)
* [OSMDeepOD](https://github.com/geometalab/OSMDeepOD) ->  perform object detection with retinanet
* [Match Bing Map Aerial Imagery with OpenStreetMap roads](https://github.com/whywww/Aerial-Imagery-and-OpenStreetMap-Retrieval)
* [Computer Vision With OpenStreetMap and SpaceNet — A Comparison](https://medium.com/the-downlinq/computer-vision-with-openstreetmap-and-spacenet-a-comparison-cc70353d0ace)
* [url-map](https://simonwillison.net/2022/Jun/12/url-map/) -> A tiny web app to create images from OpenStreetMap maps
* [Label Maker](https://github.com/developmentseed/label-maker) -> a library for creating machine-learning ready data by pairing satellite images with OpenStreetMap (OSM) vector data
* [baremaps](https://github.com/baremaps/baremaps) -> Create custom vector tiles from OpenStreetMap and other data sources with Postgis and Java.
* [osm2streets](https://github.com/a-b-street/osm2streets) -> Convert OSM to street networks with detailed geometry

## QGIS
A popular open source alternative to ArcGIS, QGIS is a desktop appication written in python and extended with plugins which are essentially python scripts
* [QGIS](https://qgis.org/en/site/)
* Create, edit, visualise, analyse and publish geospatial information. Open source alternative to ArcGIS.
* [Python scripting](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console)
* Create your own plugins using the [QGIS Plugin Builder](http://g-sherman.github.io/Qgis-Plugin-Builder/)
* [DeepLearningTools plugin](https://plugins.qgis.org/plugins/DeepLearningTools/) -> aid training Deep Learning Models
* [Mapflow.ai plugin](https://www.gislounge.com/run-ai-mapping-in-qgis-over-high-resolution-satellite-imagery/) -> various models to extract building footprints etc from Maxar imagery
* [dzetsaka plugin](https://github.com/nkarasiak/dzetsaka) -> classify different kind of vegetation
* [Coregistration-Qgis-processing](https://github.com/SMByC/Coregistration-Qgis-processing) -> Qgis processing plugin for image co-registration; projection and pixel alignment based on a target image, uses Arosics
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [buildseg](https://github.com/deepbands/buildseg) -> a building extraction plugin of QGIS based on ONNX
* [deep-learning-datasets-maker](https://github.com/deepbands/deep-learning-datasets-maker) -> a QGIS plugin to make datasets creation easier for raster and vector data
* [Modzy-QGIS-Plugin](https://github.com/modzy/Modzy-QGIS-Plugin) -> demos Vehicle Detection model
* [kart](https://plugins.qgis.org/plugins/kart/) -> provides modern, open source, distributed version-control for geospatial and tabular datasets
* [Plugin for Landcover Classification](https://github.com/atishayjn/QGIS-Plugin) -> capable of implementing machine learning algorithms such as Random forest, SVM and CNN algorithms such as UNET through a simple GUI framework.
* [pg_tileserv])(https://github.com/CrunchyData/pg_tileserv) -> A very thin PostGIS-only tile server in Go. Takes in HTTP tile requests, executes SQL, returns MVT tiles.
* [pg_featureserv](https://github.com/CrunchyData/pg_featureserv) -> Lightweight RESTful Geospatial Feature Server for PostGIS in Go
* [osm-instance-segmentation](https://github.com/mnboos/osm-instance-segmentation) -> QGIS plugin for finding changes in vector data from orthophotos (i.e. aerial imagery) using tensorflow
* [Semi-Automatic Classification Plugin](https://github.com/semiautomaticgit/SemiAutomaticClassificationPlugin) -> supervised classification of remote sensing images, providing tools for the download, the preprocessing and postprocessing of images
* [chippy-checker-editor](https://github.com/devglobalpartners/chippy-checker-editor) -> QGIS plugin for viewing and editing labeled remote sensing images
* [qgis-plugin-deepness](https://github.com/PUTvision/qgis-plugin-deepness) -> Plugin for neural network inference in QGIS: segmentation, regression and detection
* [QGPTAgent](https://github.com/momaabna/QGPTAgent) -> plugin for QGIS that utilizes the advanced natural language processing capabilities of the OpenAI GPT model to automate various processes in QGIS
* [EO Time Series Viewer](https://eo-time-series-viewer.readthedocs.io/en/latest/) -> QGIS Plugin to visualize and label raster-based earth observation time series data

## Parallel processing with Dask
Dask provides advanced parallelism and distributed out-of-core computation with a `dask.dataframe` module designed to scale pandas.
* [Dask](https://docs.dask.org/en/latest/) -> works with your PyData libraries to provide performance at scale
* [Coiled](https://coiled.io) -> a managed Dask service
* [Dask with PyTorch for large scale image analysis](https://blog.dask.org/2021/03/29/apply-pretrained-pytorch-model)
* [dask-geopandas](https://github.com/geopandas/dask-geopandas) -> offers geospatial capabilities of GeoPandas backed by Dask
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [dask-geomodeling](https://github.com/nens/dask-geomodeling) -> On-the-fly operations on geographical maps
* [dask-image](https://github.com/dask/dask-image) -> many SciPy ndimage functions implemented
* [Detecting Green Roofs in Toronto](https://toarches.medium.com/geospatial-big-data-processing-with-python-detecting-green-roofs-in-toronto-bd7bf08900f2) -> compares deep learning (Mask R-CNN & fast.ai) and classical approach using NDVI scaled on Dask
* [Analyze terabyte-scale geospatial datasets with Dask and Jupyter on AWS](https://aws.amazon.com/blogs/publicsector/analyze-terabyte-scale-geospatial-datasets-with-dask-and-jupyter-on-aws/)
* [austin-ml-change-detection-demo](https://github.com/makepath/austin-ml-change-detection-demo) -> A change detection demo for the Austin area using a pre-trained PyTorch model scaled with Dask on Planet imagery

## Jupyter
The [Jupyter](https://jupyter.org/) Notebook is a web-based interactive computing platform. There are many extensions which make it a powerful environment for analysing satellite imagery
* [jupyterlite](https://jupyterlite.readthedocs.io/en/latest/) -> JupyterLite is a JupyterLab distribution that runs entirely in the browser
* [jupyter_compare_view](https://github.com/Octoframes/jupyter_compare_view) -> Blend Between Multiple Images
* [folium](https://python-visualization.github.io/folium/quickstart.html) -> display interactive maps in Jupyter notebooks
* [ipyannotations](https://github.com/janfreyberg/ipyannotations) -> Image annotations in python using jupyter notebooks
* [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) -> create custom image classification annotators within Jupyter notebooks
* [jupyter-innotater](https://github.com/ideonate/jupyter-innotater) -> Inline data annotator for Jupyter notebooks
* [jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget) -> A Jupyter widget for annotating images with bounding boxes
* [mapboxgl-jupyter](https://github.com/mapbox/mapboxgl-jupyter) -> Use Mapbox GL JS to visualize data in a Python Jupyter notebook
* [pylabel](https://github.com/pylabel-project/pylabel) -> includes an image labeling tool that runs in a Jupyter notebook that can annotate images manually or perform automatic labeling using a pre-trained model
* [jupyterlab-s3-browser](https://github.com/IBM/jupyterlab-s3-browser) -> extension for browsing S3-compatible object storage
* [papermill](https://github.com/nteract/papermill) -> Parameterize, execute, and analyze notebooks
* [pretty-jupyter](https://github.com/JanPalasek/pretty-jupyter) -> Creates dynamic html report from jupyter notebook

## Julia language
[Julia](https://julialang.org/) looks and feels a lot like Python, but can be much faster. Julia can call Python, C, and Fortran libraries and is capabale of C/Fortran speeds. Julia can be used in the familiar Jupyterlab notebook environment
* [Why you should invest in Julia now, as a Data Scientist](https://medium.com/@logankilpatrick/why-you-should-invest-in-julia-now-as-a-data-scientist-30dc346d62e4)
* [eBook: Introduction to Datascience with Julia](https://datascience-book.gitlab.io/)
* [FastAI.jl](https://github.com/FluxML/FastAI.jl) -> Repository of best practices for deep learning in Julia, inspired by fastai
* [Flux.jl](https://github.com/FluxML/Flux.jl) -> the ML library that doesn't make you tensor. Checkout [The Deep Learning with Julia book](https://github.com/logankilpatrick/DeepLearningWithJulia)
* [GDAL.jl](https://github.com/JuliaGeo/GDAL.jl) -> Thin Julia wrapper for GDAL
* [GeoInterface.jl](https://github.com/JuliaGeo/GeoInterface.jl) -> A Julia Protocol for Geospatial Data
* [GeoJSON.jl](https://github.com/JuliaGeo/GeoJSON.jl) -> Utilities for working with GeoJSON data
* [JuliaImages: image processing and machine vision for Julia](https://juliaimages.org/stable/)
* [Julia_Geospatial](https://github.com/acgeospatial/Julia_Geospatial) -> Examples for a blog series on Geospatial Julia using ArchGDAL
* [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) -> A Julia machine learning framework
* [Proj4.jl](https://github.com/JuliaGeo/Proj4.jl) -> Julia wrapper for the PROJ cartographic projections library
* [Rasters.jl](https://github.com/rafaqz/Rasters.jl) -> types and methods for reading, writing and manipulating rasterized spatial data including GeoTIFF and NetCDF
* [RemoteS.jl](https://github.com/GenericMappingTools/RemoteS.jl) -> Remote sensing data processing
* [SatelliteToolbox.jl](https://github.com/JuliaSpace/SatelliteToolbox.jl) -> This package contains several functions to build simulations related with satellites
* [SatelliteDynamics.jl](https://github.com/sisl/SatelliteDynamics.jl) -> a satellite dynamics modeling package
* [Sentinel.jl](https://github.com/mhudecheck/Sentinel.jl) -> library for processing ESA Sentinel 2 satellite data
* [DeepSat-Kaggle](https://github.com/athulsudheesh/DeepSat-Kaggle) -> uses Julia

## Streamlit
[Streamlit](https://streamlit.io/) is an awesome python framework for creating apps with python. These apps can be used to present ML models, and here I list resources which are EO related. Note that a component is an addon which extends Streamlits basic functionality
* [cogviewer](https://github.com/mykolakozyr/cogviewer) -> Simple Cloud Optimized GeoTIFF viewer
* [cogcreator](https://github.com/mykolakozyr/cogcreator) -> Simple Cloud Optimized GeoTIFF Creator. Generates COG from GeoTIFF files.
* [cogvalidator](https://github.com/mykolakozyr/cogvalidator) -> Simple Cloud Optimized GeoTIFF validator
* [streamlit-image-comparison](https://github.com/fcakyon/streamlit-image-comparison) -> compare images with a slider. Used in [example-app-image-comparison](https://github.com/streamlit/example-app-image-comparison)
* [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) -> Streamlit Component for rendering Folium maps
* [streamlit-keplergl](https://github.com/chrieke/streamlit-keplergl) -> Streamlit component for rendering kepler.gl maps
* [streamlit-light-leaflet](https://github.com/andfanilo/streamlit-light-leaflet) -> Streamlit quick & dirty Leaflet component that sends back coordinates on map click
* [leafmap-streamlit](https://github.com/giswqs/leafmap-streamlit) -> various examples showing how to use streamlit to: create a 3D map using Kepler.gl, create a heat map, display a GeoJSON file on a map, and add a colorbar or change the basemap on a map
* [geemap-apps](https://github.com/giswqs/geemap-apps) -> build a multi-page Earth Engine App using streamlit and geemap
* [streamlit-geospatial](https://github.com/giswqs/streamlit-geospatial) -> A multi-page streamlit app for geospatial
* [geospatial-apps](https://github.com/giswqs/geospatial-apps) -> A collection of streamlit web apps for geospatial applications
* [BirdsPyView](https://github.com/rjtavares/BirdsPyView) -> convert images to top-down view and get coordinates of objects
* [Build a useful web application in Python: Geolocating Photos](https://medium.com/spatial-data-science/build-a-useful-web-application-in-python-geolocating-photos-186122de1968) -> Step by Step tutorial using Streamlit, Exif, and Pandas
* [Wild fire detection app](https://github.com/yueureka/WildFireDetection)
* [dvc-streamlit-example](https://github.com/sicara/dvc-streamlit-example) -> how dvc and streamlit can help track model performance during R&D exploration
* [stacdiscovery](https://github.com/mykolakozyr/stacdiscovery) -> Simple STAC Catalogs discovery tool
* [SARveillance](https://github.com/MJCruickshank/SARveillance) -> Sentinel-1 SAR time series analysis for OSINT use
* [streamlit-template](https://github.com/giswqs/streamlit-template) -> A streamlit app template for geospatial applications
* [streamlit-labelstudio](https://github.com/deneland/streamlit-labelstudio) -> A Streamlit component that provides an annotation interface using the LabelStudio Frontend
* [streamlit-img-label](https://github.com/lit26/streamlit-img-label) -> a graphical image annotation tool using streamlit. Annotations are saved as XML files in PASCAL VOC format
* [Streamlit-Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) -> A secure authentication module to validate user credentials in a Streamlit application
* [prettymapp](https://github.com/chrieke/prettymapp) -> Create beautiful maps from OpenStreetMap data in a webapp
* [mapa-streamlit](https://github.com/fgebhart/mapa-streamlit) -> creating 3D-printable models of the earth surface based on mapa
* [BoulderAreaDetector](https://github.com/pszemraj/BoulderAreaDetector) -> CNN to classify whether a satellite image shows an area would be a good rock climbing spot or not, deployed to streamlit app
* [streamlit-remotetileserver](https://github.com/banesullivan/streamlit-remotetileserver) -> Easily visualize a remote raster given a URL and check if it is a valid Cloud Optimized GeoTiff (COG)
* [Streamlit_Image_Sorter](https://github.com/2320sharon/Streamlit_Image_Sorter) -> Generic Image Sorter Interface for Streamlit
* [Streamlit-Folium + Snowflake + OpenStreetMap](https://github.com/randyzwitch/streamlit-folium-snowflake-openstreetmap) -> demonstrates the power of Snowflake Geospatial data types and queries combined with Streamlit
* [observing-earth-from-space-with-streamlit](https://blog.streamlit.io/observing-earth-from-space-with-streamlit/) -> blog post on the [SatSchool](https://github.com/Spiruel/SatSchool) app
