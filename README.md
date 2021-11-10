# Fine-Needle-Aspiration (FNA) screening pipeline
Collaboration with UMass Medical Center to automatically detect follicular clusters in unstained slides.  
This repository includes code for running our Slide Scanner and FNA-Net deep learning pipeline.


# Software Requirements
* Python v3.8.10

# Installation
### How to import rasterio (instructions for Windows 10)
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
* download the MARS-Net from the Github repository https://github.com/kleelab-bch/MARS-Net
* Follow the instructions in MARS-Net repository for installation and cropping. 
* Then, run the following files in order to train and evaluate the classifier
    * Segmentation/models/train.py 
    * Segmentation/models/predict.py 
    * Segmentation/models/evaluate_classifier.py 

# Training and Evaluation of the Faster R-CNN,
* please refer to Tensorflow Object Detection API at https://github.com/tensorflow/models/tree/master/research/object_detection

# Evaluation of the box overlaps between ground truth and the predictions
This section can replicate the figures that are presented in the paper.
* Download FNA-Net from this repository
* Run in cmd or terminal based on the user's operating systems (Windows 10 / Ubuntu 16.04)
  * >evaluation/run_eval.py
  * Results will be generated in the evaluation/generated folder
* Inside run_eval_final function in the run_eval.py, 
  * overlay_two_model_overlapped_polygons_over_images function draws bounding box for every follicular cluster classified to be true in the image patch
  * bootstrap_two_model_polygons function bootstraps the samples. This can take several minutes.
  * After bootstrapping, bootstrap_analysis function performs analysis such as plotting histogram, and precision recall curves      
  * bootstrap_analysis_compare_precision_recall can be run at the end after bootstrapping samples from each model (MTL, faster R-CNN and MTL+faster R-CNN). It draws precision-recall curve for each model on the same plot for comparison.

