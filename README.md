# Cat-Vs-Dog-Classification Using CNN

- Prediction on cat and dogs data to classify them in right class using CNN.
- Testing loss: 0.02% 
- Testing Accuracy: 99%
- Data Set used:  https://www.kaggle.com/datasets/salader/dogs-vs-cats
____________

## Overview
This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN). The dataset used is the "Dogs vs Cats" dataset from Kaggle. The model achieves high accuracy, making it effective for distinguishing between images of cats and dogs.

### Key Metrics:
- **Testing Accuracy:** 99%
- **Testing Loss:** 0.02%

## Dataset
The dataset used in this project is the "Dogs vs Cats" dataset, which contains 25,000 labeled images of cats and dogs. It is available on Kaggle:
- **Dataset URL:** [Dogs vs Cats on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## Data Preprocessing
1. **Downloading the dataset:**
   - The dataset is downloaded directly from Kaggle using the following commands:
   ```bash
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !kaggle datasets download -d salader/dogs-vs-cats
   ```

2. **Extracting the dataset:**
   - The downloaded `.zip` file is unzipped and the images are stored in separate directories for training and testing.

3. **Data Normalization:**
   - Images are resized to 256x256 pixels and pixel values are normalized by dividing by 255 to scale them between 0 and 1.

4. **Image Loading and Batching:**
   - The dataset is loaded using TensorFlow’s `image_dataset_from_directory`, which efficiently loads and batches images in real-time using generators.

## Model Architecture

The CNN model architecture is as follows:
- **Conv2D Layers:** 
  - 3 convolutional layers with 32, 64, and 128 filters, respectively.
  - Each convolutional layer is followed by a MaxPooling layer.
- **Fully Connected Layers:**
  - 3 dense layers (128, 64, and 1 unit).
  - The final layer uses a sigmoid activation function for binary classification (cat or dog).
  
The architecture summary is:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Conv2D (Conv2D)              (None, 254, 254, 32)      896
MaxPooling2D (MaxPooling2D)  (None, 127, 127, 32)      0
Conv2D (Conv2D)              (None, 125, 125, 64)      18496
MaxPooling2D (MaxPooling2D)  (None, 62, 62, 64)        0
Conv2D (Conv2D)              (None, 60, 60, 128)       73856
MaxPooling2D (MaxPooling2D)  (None, 30, 30, 128)       0
Flatten (Flatten)            (None, 115200)            0
Dense (Dense)                (None, 128)               14745728
Dense (Dense)                (None, 64)                8256
Dense (Dense)                (None, 1)                 65
=================================================================
Total params: 14,847,297
Trainable params: 14,847,297
```

## Model Training

The model is trained for 10 epochs using the Adam optimizer and binary cross-entropy loss function. The performance on both the training and validation datasets is monitored.

### Training Results:
- **Epoch 1:** Accuracy = 55.91%, Validation Accuracy = 63.06%
- **Epoch 10:** Accuracy = 99.26%, Validation Accuracy = 76.10%

### Insights:
- The model performs well on the training data but shows some overfitting, as the validation accuracy is lower than the training accuracy.
- Overfitting can be mitigated by adding more data, data augmentation, regularization, and dropout layers.

## Visualizations

1. **Accuracy Plot:**
   - The training and validation accuracy are plotted, showing an increase in accuracy over epochs. However, there is a gap indicating potential overfitting.

2. **Loss Plot:**
   - The training loss decreases over time, while the validation loss increases, confirming the overfitting issue.

## Evaluation on New Data

The trained model was tested on 15 new images:
- **Cat Images:** 10 out of 15 were correctly classified as cats.
- **Dog Images:** 13 out of 15 were correctly classified as dogs.

This shows that the model is performing well in classifying new, unseen images.

## Conclusion

This CNN-based model is effective for classifying images of cats and dogs, achieving high accuracy on both training and testing datasets. However, steps such as data augmentation and regularization could be applied to further improve generalization and reduce overfitting.
   ```
