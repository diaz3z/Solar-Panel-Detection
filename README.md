
# Solar Panel Detection Using CNN

This project implements a Convolutional Neural Network (CNN) model to detect the presence of solar panels in satellite images. The model is trained on labeled image data and aims to classify images as either containing solar panels or not.

# Introduction
The rapid growth in solar energy adoption has increased the need for automated methods to identify solar panel installations. This project addresses this challenge by leveraging deep learning to classify satellite images. The CNN model is trained on a dataset of images labeled as either containing solar panels or not.
# Dataset
The dataset consists of satellite images with corresponding labels indicating the presence or absence of solar panels. The data is divided into training and validation sets, and the images are processed to a uniform size of 101x101x3.
```bash
  - training/
  - 0.tif
  - 1.tif
  - ...
- labels_training.csv

```
# Model Architecture
The CNN model consists of the following layers:

•Convolutional layers with ReLU activation and Batch Normalization

•MaxPooling layers to downsample feature maps

•Global MaxPooling before the output layer

•Dense output layer with a sigmoid activation function
# Model Summary:
```bash
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 99, 99, 32)        896       
 batch_normalization (BatchNormalization)  (None, 99, 99, 32)        128       
 conv2d_1 (Conv2D)           (None, 97, 97, 64)        18496      
 ...
 global_max_pooling2d (GlobalMaxPooling2D)  (None, 128)               0         
 dense (Dense)               (None, 1)                 129       
=================================================================
Total params: 391105
Trainable params: 389889
Non-trainable params: 1216


```
# Training and Evaluation
The model is trained using binary cross-entropy loss and the Adam optimizer. Stratified K-Fold cross-validation is employed to assess the model's performance across different splits of the dataset.

During training, class weights are applied to handle the imbalance in the dataset (with more images not containing solar panels than those that do).


# Cross-Validation Results:
The model achieves high accuracy and demonstrates strong performance across various metrics.


# Confusion Matrix:
The following images illustrate examples of true positives, true negatives, false positives, and false negatives:

•True Positives: Images correctly classified as containing solar panels.

•False Positives: Images incorrectly classified as containing solar panels.

•True Negatives: Images correctly classified as not containing solar panels.

•False Negatives: Images incorrectly classified as not containing solar panels.

# Screenshot
![download](https://github.com/user-attachments/assets/b776388a-a789-417c-a891-d051dff434a6)



# Installation



Clone the project

```bash
git clone https://github.com/diaz3z/Solar-Panel-Detection.git

```

Go to the project directory

```bash
  cd Solar-Panel-Detection
```

Install the required libraries:

```bash
    pip install tensorflow
    pip install torch torchvision
    pip install keras
    pip install pandas
    pip install opencv-python
    pip install numpy
    pip install matplotlib
    pip install scikit-learn
    pip install seaborn
    pip install albumentations
    pip install glob2
    pip install tqdm
    pip install pillow

```

Run the training notebook:

## License
This project is licensed under the MIT License - see the LICENSE file for details.
