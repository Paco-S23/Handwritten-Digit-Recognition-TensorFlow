# 🤖 Handwritten Digit Recognition with Neural Networks and TensorFlow

This project is an implementation of a **Deep Learning** model to classify images of handwritten digits (`0-9`).

---

## 🎯 Project Objective

The main goal is to build a **simple yet effective image classifier**, demonstrating a complete Deep Learning workflow:

-   Loading and preparing image data.
-   Building a Neural Network architecture.
-   Training the model.
-   Evaluating its performance.

---

## 📂 Dataset

-   **Name:** MNIST Handwritten Digit Database
-   **Source:** `tensorflow.keras.datasets.mnist`
-   **Size:** 70,000 images in total.
    -   60,000 for training.
    -   10,000 for testing.
-   **Format:** 28x28 grayscale images.
-   **Labels:** Digits from `0` to `9`.

---

## ⚙️ Methodology

1.  **Data Loading**
    The **TensorFlow** library was used to load and explore the `MNIST` dataset, which comes pre-split.

2.  **Image Preprocessing**
    **Normalization** was applied to the pixel values by dividing them by `255.0`. This scales the input data to the `[0, 1]` range to improve training performance and stability.

3.  **Data Splitting**
    The dataset already includes a standard split:
    -   **Training Set (60,000 images)** → The model learns the patterns.
    -   **Test Set (10,000 images)** → Used for the final performance evaluation and as a validation set during training.

4.  **Model Building and Training**
    -   A **Sequential Neural Network** from Keras was used, which is ideal for image classification problems like this one.
    -   **Architecture:** `Flatten` → `Dense(128, 'relu')` → `Dense(10, 'softmax')`.

5.  **Evaluation**
    -   The **accuracy** was measured on the test set to evaluate the model's ability to generalize.

---

## 📊 Results

-   **Test Accuracy:** `97.60%`
-   The model demonstrates high effectiveness in correctly differentiating and classifying the ten handwritten digits.

---

## 💻 Example Code Snippet

```python
import tensorflow as tf

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
