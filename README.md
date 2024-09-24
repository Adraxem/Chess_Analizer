# Chess Piece Classification with CNN

This repository contains a convolutional neural network (CNN) implementation for classifying chess pieces based on images. The dataset consists of chess piece images stored in separate class folders. The model is trained using different batch sizes and epochs to compare validation accuracy across configurations.

### Key Features
- **Image Preprocessing**:
  - Images are loaded from the `chessdir/` directory.
  - Each image is resized to 96x96 pixels.
  
- **Convolutional Neural Network (CNN) Architecture**:
  - The CNN consists of three convolutional layers followed by max-pooling layers.
  - The network uses **ReLU** activations and a fully connected layer to predict the class probabilities.
  - The final layer uses **softmax** activation for multi-class classification.

- **Training Configurations**:
  - The model is trained and evaluated using varying batch sizes (8, 32, 64, 128) and different epoch values (10, 20, 30, 50, 80).
  - Each configuration is tested to observe its impact on validation accuracy.

- **Metrics and Visualization**:
  - After training, validation accuracy is plotted against batch size and number of epochs to identify optimal training configurations.
  - Plots of **Accuracy vs. Batch Size** and **Accuracy vs. Epochs** are generated for analysis.

### Requirements
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

### Usage
1. Organize the dataset into folders inside `chessdir/`, with each folder representing a class (chess piece type).
2. Run the provided code to train the CNN on the dataset and evaluate it using different batch sizes and epoch values.
3. The results will be displayed as validation accuracy plotted against batch size and epochs.

### Output
- **Validation Accuracy** for different batch sizes (8, 32, 64, 128).
- **Validation Accuracy** for different epoch values (10, 20, 30, 50, 80).
- **Visualization** of results with two plots:
  - Accuracy vs. Batch Size
  - Accuracy vs. Epochs

### License
This project is licensed under the MIT License.
