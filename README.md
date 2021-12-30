# Fine-Needle-Aspiration screening pipeline (FNA-Net)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=plastic)](https://opensource.org/licenses/MIT) 
[![Repo Size](https://img.shields.io/github/repo-size/kleelab-bch/FNA-Net?style=plastic)]()

**Screening Adequacy of Unstained Fine Needle Aspiration Samples
Using a Deep Learning-based Classifier**  
by Junbong Jang, Young Kim, Brian Westgate, Yang Zong, Caleb Hallinan, Ali Akalin, Kwonmoo Lee

<!-- To learn more about our pipeline (FNA-Net), please read the [paper]( ) -->

<div text-align="center">
  <img width="300" src="./assets/cover_FNA.png" alt="FNA-Net">
</div>  

Collaboration with UMass Medical Center to automatically detect follicular clusters in unstained slides.  
This repository includes code for running our Slide Scanner and FNA-Net deep learning pipeline.


# Software Requirements
* Works in Ubuntu v16.04, v18.04, Windows 10, and Mac OS.
* Python v3.8
* Tensorflow v2.2 & CUDA v10.1 or Tensorflow v2.4 & CUDA v11.3


# Installation
* install Python and Python packages as listed in requirements.txt

#### How to import rasterio (instructions for Windows 10)
* Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/
* Download 
  * Fiona-1.8.19-cp38-cp38-win_amd64
  * GDAL-3.2.3-cp38-cp38-win_amd64
  * rasterio-1.2.3-cp38-cp38-win_amd64
* Install them in the following order
  * pip install Fiona-1.8.19-cp38-cp38-win_amd64
  * pip install GDAL-3.2.3-cp38-cp38-win_amd64
  * pip install rasterio-1.2.3-cp38-cp38-win_amd64
  
# Training and Evaluation of the patch-wise classifier,
* Download MARS-Net from the Github repository https://github.com/kleelab-bch/MARS-Net
* Follow the instructions in MARS-Net repository for installation and cropping. 
* Then, run the following Python scripts in order to train and evaluate the classifier
    * MARS-Net/models/train.py 
    * MARS-Net/models/predict.py 
    * MARS-Net/models/evaluate_classifier.py 

# Training of the Faster R-CNN,
* Please refer to documents for Tensorflow Object Detection API at https://github.com/tensorflow/models/tree/master/research/object_detection
* We used "Faster R-CNN Inception ResNet V2 640x640" downloaded from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

# Evaluation of the FNA-Net
This section calculates the area overlap between the ground truth mask and the prediction boxes and
visualizes follicular cluster detection results per image. 
Also, it performs the hierarchical bootstrapping and visualizes its summary statistics and precision-recall curves of models.

* Download and install this repository 
* In the command prompt or terminal based on the user's operating systems, run  
  * \> python evaluation/run_eval.py
  * Results will be generated in the evaluation/generated folder
* Inside run_eval_final function in the run_eval.py, 
  * overlay_two_model_overlapped_polygons_over_images function draws bounding box for every follicular cluster classified to be true in the image patch
  * bootstrap_two_model_polygons function bootstraps the samples. This can take several minutes.
  * After bootstrapping, bootstrap_analysis function performs analysis such as plotting histogram and precision-recall curves      
  * bootstrap_analysis_compare_precision_recall can be run at the end after bootstrapping samples from each model (MTL, faster R-CNN, and MTL+faster R-CNN). It draws a precision-recall curve for each model on the same plot for comparison.
