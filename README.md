# QMNIST Classification Using a Simple Neural Network

This project demonstrates the use of a simple neural network to classify images from the QMNIST dataset, an extended version of the MNIST dataset. It covers dataset preparation, image processing, model design and training, and model evaluation.

---

## 1. Dataset Introduction

### Dataset Source
The dataset used in this project is the **QMNIST - The Extended MNIST Dataset**, sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/qmnist-the-extended-mnist-dataset-120k-images). It contains **120,000 images** of handwritten digits (0-9), making it an expanded version of the classic MNIST dataset.

### Dataset Split
The dataset is split into training and testing sets using the following code:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
```

- `test_size=0.2` means 20% of the data is used for testing and 80% for training.
- `random_state=42` ensures reproducibility.

---

## 2. Image Processing

### Image Processing Pipeline
The images are preprocessed as follows:
- Pixel values are scaled from the range `[0, 255]` to `[0, 1]` for normalization.

Relevant code:

```python
# Scaling the values from [0, 255] to [0, 1] (normalization)
X_train = X_train / 255
X_test = X_test / 255
print(X_train[10])
```

**Explanation**:
- Normalization ensures that the pixel values are in a consistent range, which helps the neural network converge faster during training.

---

## 3. Model Introduction

### Model Overview
The model is a simple feedforward neural network with the following architecture:
1. **Input Layer**: Flattens the input image of size `(28, 28)` into a 1D vector.
2. **Hidden Layers**:
   - A dense layer with 128 neurons and ReLU activation.
   - A dense layer with 64 neurons and ReLU activation.
3. **Output Layer**: A dense layer with 10 neurons (one for each class) and softmax activation for multiclass classification.

### Model Code
```python
import tensorflow as tf
from tensorflow import keras

# Setting up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
    keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
    keras.layers.Dense(64, activation='relu'),   # Hidden layer 2
    keras.layers.Dense(10, activation='softmax') # Output layer
])

# Compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Neural Network
model.fit(X_train, Y_train, epochs=10)
```

### Training Log
```
Epoch 1/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 14s 4ms/step - accuracy: 0.9002 - loss: 0.3394
Epoch 2/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 21s 5ms/step - accuracy: 0.9708 - loss: 0.0959
Epoch 3/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 13s 4ms/step - accuracy: 0.9801 - loss: 0.0651
Epoch 4/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 22s 5ms/step - accuracy: 0.9853 - loss: 0.0475
Epoch 5/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 19s 4ms/step - accuracy: 0.9877 - loss: 0.0384
Epoch 6/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 13s 4ms/step - accuracy: 0.9895 - loss: 0.0317
Epoch 7/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9910 - loss: 0.0263
Epoch 8/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 13s 4ms/step - accuracy: 0.9918 - loss: 0.0230
Epoch 9/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 13s 4ms/step - accuracy: 0.9930 - loss: 0.0210
Epoch 10/10
3000/3000 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - accuracy: 0.9931 - loss: 0.0198
```

### Test Data Evaluation
```
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.9770 - loss: 0.1230
```

### Confusion Matrix

<img width="1123" height="602" alt="image" src="https://github.com/user-attachments/assets/ef449984-1e56-447d-8996-0c96e2fc57e4" />


---

## 4. Summary and Training Effect Analysis

- The model achieves a **training accuracy of 99.31%** and a **test accuracy of 97.70%**, demonstrating strong performance on the QMNIST dataset.
- The training loss decreases steadily over epochs, indicating good convergence.
- The test accuracy is slightly lower than the training accuracy, suggesting minor overfitting. This can be addressed by adding regularization techniques (e.g., dropout) or using data augmentation.
- The confusion matrix (to be inserted) will provide insights into which classes the model struggles with.

---

## 5. References

- [QMNIST Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/qmnist-the-extended-mnist-dataset-120k-images)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

---
> **All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.**

---

Let me know if you need further adjustments or additional details!
