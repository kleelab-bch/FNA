# Fine-Needle-Aspiration
Collaboration with UMass Medical Center to automatically detect follicular clusters in unstained slides.  
This repository includes code for running our Slide Scanner and FNA-Net deep learning pipeline.

For Training and evaluating image patch classification,
* MARS-Net pipeline at https://github.com/kleelab-bch/MARS-Net
    * Segmentation/models/train.py 
    * Segmentation/models/predict.py 
    * Segmentation/models/evaluate_classifier.py 

For Evaluation (instructions for Windows 10)
* Draw bounding box for every follicular cluster classified to be true in the image patch
    * evaluation/run_eval.py
    * see results in evaluation/generated folder

In order to import rasterio,
* Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/
* Download 
  * Fiona-1.8.19-cp38-cp38-win_amd64
  * GDAL-3.2.3-cp38-cp38-win_amd64
  * rasterio-1.2.3-cp38-cp38-win_amd64
* Install them in the following order
  * pip install Fiona-1.8.19-cp38-cp38-win_amd64
  * pip install GDAL-3.2.3-cp38-cp38-win_amd64
  * pip install rasterio-1.2.3-cp38-cp38-win_amd64