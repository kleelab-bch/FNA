# Fine-Needle-Aspiration
Collaboration with UMass Medical Center to automatically detect follicular clusters in unstained slides.  
This repository includes code for Low-cost Slide Scanner and FNA-Net deep learning pipeline

How To
* MARS-Net pipelline to train a new model and evaluate image patch classification
    * Segmentation/models/train.py 
    * Segmentation/models/predict.py 
    * Segmentation/models/evaluate_classifier.py
* Draw bounding box for every follicular cluster classified to be true in the image patch
    * evaluation/run_eval.py
    * see results in evaluation/generated folder