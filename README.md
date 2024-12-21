Multi-Face Detection Based on Gender and Age
============================================

This project implements a multi-face detection system that predicts gender and age based on two datasets: [FlickrFacesHQ Dataset (FFHQ)](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq/data) and [UTKFace](https://susanqq.github.io/UTKFace/). The system utilizes two models with different training approaches: one with frozen layers and another with unfrozen layers.


## Requirements

Ensure you have all the necessary dependencies installed before running the scripts. Use the `requirements.txt` file to install them.

### Installing Dependencies

Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

Model Architectures
-------------------

### Unfrozen Model

1.  **Base Model**: MobileNetV2 (pre-trained on ImageNet).
    
    *   Input shape: (128, 128, 3).
        
    *   Include top: False.
        
    *   Weights: Pre-trained on ImageNet.
        
2.  **Modifications**:
    
    *   **Global Pooling**: GlobalAveragePooling2D.
        
    *   **Batch Normalization**: Added after pooling.
        
    *   **Dropout**: 0.5.
        
    *   **Age Output**: Dense layer with linear activation for regression.
        
    *   **Gender Output**: Dense layer with sigmoid activation for binary classification.
        
3.  **Training Configuration**:
    
    *   Base model layers: Trainable.
        
    *   Loss functions: Mean Squared Error (age), Binary Crossentropy (gender).
        
    *   Optimizer: Adam (learning rate = 1e-4).
        
    *   Metrics: Mean Absolute Error (age), Accuracy (gender).
        

### Frozen Model

1.  **Base Model**: MobileNetV2 (pre-trained on ImageNet).
    
    *   Input shape: (128, 128, 3).
        
    *   Include top: False.
        
    *   Weights: Pre-trained on ImageNet.
        
2.  **Modifications**:
    
    *   **Global Pooling**: GlobalAveragePooling2D.
        
    *   **Dropout**: 0.5.
        
    *   **Age Output**: Dense layer with linear activation for regression.
        
    *   **Gender Output**: Dense layer with sigmoid activation for binary classification.
        
3.  **Training Configuration**:
    
    *   Base model layers: Frozen.
        
    *   Loss functions: Mean Squared Error (age), Binary Crossentropy (gender).
        
    *   Optimizer: Adam.
        
    *   Metrics: Mean Absolute Error (age), Accuracy (gender).
        

Dataset Description
-------------------

1.  **FlickrFacesHQ Dataset (FFHQ)**:
    
    *   High-quality images of faces.
        
    *   Suitable for high-resolution tasks.
        
2.  **UTKFace Dataset**:
    
    *   Contains images with annotated age, gender, and ethnicity.
        
    *   Suitable for age and gender prediction tasks.
        

Model Files
-----------

*   model/freeze\_model.h5: Trained model with frozen layers.
    
*   model/unfreeze\_model.h5: Trained model with unfrozen layers.
    

Scripts
-------

1.  **Model Build Script**: Model\_Build.ipynb
    
    *   Jupyter Notebook used for building and training the models.
        
2.  **Test Script**: test.py
    
    *   Python script for evaluating the trained models.
        

Usage Instructions
------------------

### Training the Model

1.  Use Model\_Build.ipynb to train the model.
    
2.  Ensure datasets are downloaded and paths are correctly set.
    

### Testing the Model

1.  Run test.py to evaluate the trained models.
    
2.  Specify the model file (freeze\_model.h5 or unfreeze\_model.h5) in the script.
    

## Results

| **Metric**              | **Frozen Model**   | **Unfrozen Model** |
|--------------------------|--------------------|--------------------|
| **Training Loss**        | 123.45            | 30.91             |
| **Validation Loss**      | 290.09            | 103.39            |
| **Age Output MAE (Val)** | 12.90             | 7.34              |
| **Gender Accuracy (Val)**| 55.40%            | 81.67%            |


Conclusion
----------

The unfrozen model significantly outperforms the frozen model in both age prediction and gender classification. The higher accuracy and lower mean absolute error (MAE) demonstrate the benefits of fine-tuning all layers during training.
