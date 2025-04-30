<div align="center">
  <p>
    <a href="https://www.satellite-image-deep-learning.com/">
        <img src="logo.png" width="700">
    </a>
</p>
  <h2>Software for working with satellite & aerial imagery ML datasets.</h2>

# 👉 [satellite-image-deep-learning.com](https://www.satellite-image-deep-learning.com/) 👈

</div>

**How to use this repository:** if you know exactly what you are looking for (e.g. you have the paper name) you can `Control+F` to search for it in this page (or search in the raw markdown).

## Contents
* [Deep learning frameworks](https://github.com/satellite-image-deep-learning/software#deep-learning-frameworks)
* [Image dataset creation](https://github.com/satellite-image-deep-learning/software#image-dataset-creation)
* [Image chipping/tiling & merging predictions](https://github.com/satellite-image-deep-learning/software#image-chippingtiling--merging-predictions)
* [Image processing, handling, manipulation](https://github.com/satellite-image-deep-learning/software#image-processing-handling-manipulation)
* [Image augmentation packages](https://github.com/satellite-image-deep-learning/software#image-augmentation-packages)
* [SpatioTemporal Asset Catalog specification (STAC)](https://github.com/satellite-image-deep-learning/software#spatiotemporal-asset-catalog-specification-stac)
* [OpenStreetMap](https://github.com/satellite-image-deep-learning/software#openstreetmap)
* [QGIS](https://github.com/satellite-image-deep-learning/software#qgis)
* [Jupyter](https://github.com/satellite-image-deep-learning/software#jupyter)
* [Streamlit](https://github.com/satellite-image-deep-learning/software#streamlit)

# Deep learning frameworks
* [TorchGeo](https://github.com/microsoft/torchgeo) -> PyTorch library providing datasets, samplers, transforms, and pre-trained models specific to geospatial data. 📺 YouTube: [TorchGeo with Caleb Robinson](https://youtu.be/ET8Hb_HqNJQ)
* [rastervision](https://rastervision.io/) -> An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets. 📺 YouTube: [Raster Vision with Adeel Hassan](https://youtu.be/hH59fQ-HhZg)
* [rslearn](https://github.com/allenai/rslearn) -> from Allenai, a tool for developing remote sensing datasets and models.
* [segmentation_gym](https://github.com/Doodleverse/segmentation_gym) -> A neural gym for training deep learning models to carry out geoscientific image segmentation, uses keras. 📺 YouTube: [Satellite image segmentation using the Doodleverse segmentation gym with Dan Buscombe](https://youtu.be/0I1TOOGfdZ0)
* [GeoDeep](https://github.com/uav4geo/GeoDeep) -> Free and open source library for AI object detection and semantic segmentation in geospatial rasters.
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

# Image dataset creation
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
* [geetiles](https://github.com/rramosp/geetiles) -> download Google Earth Engine datasets to tiles as geotiff arrays

# Image chipping/tiling & merging predictions
Due to the large size of raw images, it is often necessary to chip or tile them into smaller patches before annotation and training. Similarly, models typically make predictions on these smaller patches, and it is essential to merge these predictions to reconstruct the full-sized image accurately.
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
* [SAHI](https://github.com/obss/sahi) -> Utilties for performing sliced inference and generating merged predictions. Also checkout [supervision](https://supervision.roboflow.com/latest/) which additionally supports rotated bounding boxes.
* [geo2ml](https://github.com/mayrajeo/geo2ml) -> Python library and module for converting earth observation data to be suitable for machine learning models, Converting vector data to COCO and YOLO formats and creating required dataset files, Rasterizing polygon geometries for semantic segmentation tasks, Tiling larger rasters and shapefiles into smaller patches
* [FlipnSlide](https://github.com/elliesch/flipnslide) -> a concise tiling and augmentation strategy to prepare large scientific images for use with GPU-enabled algorithms. Outputs PyTorch-ready preprocessed datasets from a single large image.
* [segmenteverygrain](https://github.com/zsylvester/segmenteverygrain) -> implements merging of segmentation predictions with patching and weighted merging in `predict_big_image`
* [image-bbox-slicer](https://image-bbox-slicer.readthedocs.io/en/latest/) -> slice images and their bounding box annotations into smaller tiles, both into specific sizes and into any arbitrary number of equal parts
* [stacchip](https://github.com/Clay-foundation/stacchip) -> Dynamically create image chips from STAC items

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
* [AdverseWeatherSimulation](https://github.com/RicardooYoung/AdverseWeatherSimulation) -> a simulator that generates foggy, rainy, smoky and cloudy image over a clear remote sensing image

# SpatioTemporal Asset Catalog specification (STAC)
The STAC specification provides a common metadata specification, API, and catalog format to describe geospatial assets, so they can more easily indexed and discovered.
* Spec at https://github.com/radiantearth/stac-spec
* [STAC 1.0.0: The State of the STAC Software Ecosystem](https://medium.com/radiant-earth-insights/stac-1-0-0-software-ecosystem-updates-da4e800a4973)
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
* [STAC Machine Learning Model (MLM) Extension](https://github.com/stac-extensions/mlm) -> to describe ML models, their training details, and inference runtime requirements.
* [STAC Label Extension](https://github.com/stac-extensions/label) -> facilitates the integration of labeled Areas of Interest (AOIs) with the STAC for machine learning and other applications.

# OpenStreetMap
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
* [ohsome-planet](https://github.com/GIScience/ohsome-planet) -> Transform OSM (history) PBF files into GeoParquet. Enrich with OSM changeset metadata and country information.

# QGIS
[QGIS](https://qgis.org/en/site/) is a desktop appication written in python and extended with plugins which are essentially python scripts
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
* [Deepness](https://github.com/PUTvision/qgis-plugin-deepness) -> a remote sensing plugin that enables deep learning inference in QGIS
* [jupyter-remote-qgis-proxy](https://github.com/sunu/jupyter-remote-qgis-proxy) -> Run QGIS inside a remote Desktop on Jupyter and open remote data sources

# Jupyter
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
* [Fast Dash is an innovative way to deploy your Python code as interactive web apps with minimal changes](https://github.com/dkedar7/fast_dash)

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
* [Streamlit-Folium + Snowflake + OpenStreetMap](https://github.com/cuulee/streamlit-folium-snowflake-openstreetmap) -> demonstrates the power of Snowflake Geospatial data types and queries combined with Streamlit
* [observing-earth-from-space-with-streamlit](https://blog.streamlit.io/observing-earth-from-space-with-streamlit/) -> blog post on the [SatSchool](https://github.com/Spiruel/SatSchool) app

## Gradio
* [gradio_folium](https://github.com/freddyaboulton/gradio_folium) -> Display Interactive Maps Created with Folium, [example](https://huggingface.co/spaces/freddyaboulton/gradio_folium)
