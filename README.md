# CNN Sketched Character Detection Model

This project is a deep learning-based model for detecting sketched characters in images. The model is built using the Keras library and utilizes the ResNet50 architecture as the base model. Additionally, the model employs the Generalized Intersection over Union (GIoU) as the loss function.

## Dependencies

The following libraries are used in this project:

- Keras
- Scikit-learn
- TensorFlow
- NumPy

## Model Architecture

The model is based on the ResNet50 architecture, which is a popular convolutional neural network (CNN) model widely used for various image recognition tasks. The model is trained on a dataset of sketched characters and is capable of detecting and localizing these characters in input images.

The model achieves an accuracy of 70% without any fine-tuning.

## Training and Evaluation

The model is trained using the Keras library, with the GIoU loss function as the objective. The training process involves preprocessing the input data, defining the model architecture, and optimizing the model parameters.

During the evaluation phase, the model's performance is assessed using relevant metrics, such as precision, recall, and F1-score.

## Usage

To use the sketched character detection model, you can follow these steps:

1. Install the required dependencies.
2. Prepare your dataset of sketched characters.
3. Preprocess the input data as required by the model.
4. Train the model using the provided scripts.
5. Evaluate the model's performance on a test set.
6. Utilize the trained model for sketched character detection in your applications.

## Acknowledgments

This project was developed as part of the COMP2211 course at the Hong Kong University of Science and Technology (HKUST) in 2024. I would like to express my gratitude to the teaching team for their guidance and support throughout the course.