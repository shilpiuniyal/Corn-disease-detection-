# Corn-disease-detection-
# Crop Disease Detection using Convolutional Neural Networks (CNN)

This project focuses on the classification of crop diseases using deep learning techniques, particularly Convolutional Neural Networks (CNNs). The model is trained to classify various corn leaf diseases using a dataset of labeled images. The dataset is divided into training, validation, and testing sets, with data augmentation applied during training.

## Project Overview

The goal of this project is to build a deep learning model that can automatically detect and classify different diseases affecting crops (specifically corn) based on images of leaves. The model uses transfer learning with the VGG16 architecture and data augmentation techniques to improve generalization and accuracy.

## Dataset

The dataset used in this project is stored on Google Drive and organized into three main directories:

- `train`: Training set containing labeled images.
- `val`: Validation set used for hyperparameter tuning and model validation.
- `test`: Test set used for evaluating the model's final performance.

The dataset includes multiple categories of crop diseases, such as:
- Corn - Cercospora Leaf Spot (Gray Leaf Spot)
- Corn - Common Rust
- Corn - Healthy
- Corn - Northern Leaf Blight

## Model Architecture

The model is built using the **VGG16** architecture with the following layers added:
- Convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Global Average Pooling layer to reduce the dimensionality of the feature maps.
- Fully connected layers for classification.
- Dropout layers to reduce overfitting.

### Key Libraries
- **TensorFlow/Keras**: For building and training the CNN model.
- **OpenCV**: For image processing.
- **Matplotlib**: For plotting training/validation accuracy and loss curves.
- **Scikit-learn**: For generating classification reports and confusion matrices.
- **NumPy**: For handling numerical data efficiently.

## Project Workflow

1. **Data Preparation**:
   - The images are preprocessed using `ImageDataGenerator`, which performs image rescaling and augmentation (rotation, zoom, flip, etc.).
   
2. **Model Training**:
   - The model is trained on the training data (`train_dir`) and validated on the validation data (`val_dir`).
   - A pre-trained **VGG16** model is used for transfer learning, with its convolutional layers frozen during initial training.
   - Data augmentation is used to avoid overfitting.

3. **Model Evaluation**:
   - The trained model is evaluated on the test set (`test_dir`) and various metrics are generated including accuracy, confusion matrix, and classification report.

4. **Prediction**:
   - The model is tested on individual images, and the class labels along with the confidence scores are printed.

## Code Breakdown

### Data Generators

The training, validation, and test datasets are loaded using `ImageDataGenerator` with various augmentations applied to the training data to improve the robustness of the model.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
