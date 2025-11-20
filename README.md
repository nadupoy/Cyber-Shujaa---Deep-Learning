# 🕸️ Cyber Shujaa - Deep Learning 🚀

This is my solution to the week 10 assignment in the Cyber Shujaa program for the 
**Data and AI Specialist** track.

## 🧭 Table of contents

- [🌟 Overview](#🌟-overview)
  - [The assignment 🎯](#the-assignment-🎯)
  - [Links 🔗](#links-🔗)
- [🛠️ My process](#🛠️-my-process)
  - [Built with 🧱](#built-with-🧱)
  - [What I learned 🧠](#what-i-learned-🧠)
  - [Continued development 🌱](#continued-development-🌱)
  - [Useful resources 📚](#useful-resources-📚)
- [👩🏽‍💻 Author](#👩🏽‍💻-author)

## 🌟 Overview

## The assignment 🎯

In this assignment, we were to apply our understanding of Artificial Neural Networks 
and TensorFlow/Keras to build, train, evaluate, and document an image classification 
model using the **MNIST dataset**.

**MNIST** stands for **Modified National Institute of Standards and Technology** 
dataset. It is a **benchmark dataset** widely used in training and testing machine 
learning and deep learning models for **handwritten digit recognition**.

| **Item**   | **Description**                            |
| ---------- | ------------------------------------------ |
| **Images** | 70,000 grayscale images of digits (0 to 9) |
| **Size**   | Each image is 28x28 pixels (784 total)     |
| **Labels** | 10 classes (digits 0 to 9)                 |
| **Split**  | 60,000 training images, 10,000 test images |

By completing the assignment, we would demonstrate our ability to:

- Preprocess and explore image data
- Design and build the ANN architecture
- Compile, train, and validate the deep learning model
- Evaluate the model on the test set and report the final test accuracy
- Visualize model training history
- Save and load trained models using the Keras format

We were to complete the following tasks:

1. Load the MNIST dataset `tensorflow.keras.datasets`.
2. Visualize at least 9 random images with their labels using `matplotlib`.
3. Normalize the pixel values [0,1] range.
4. One-hot encode the labels using to `to_categorical`.
5. Print dataset shapes and confirm preprocessing.
6. Use the `Sequential` model.
7. Include at least:
    - `Flatten` layer as input layer
    - Two `Dense` hidden layers (e.g., 128 and 64 neurons) with `ReLu` activation
    - Dropout layers (e.g., 0.3) for regularization
    - Output layer with 10 neurons and `softmax` activation
8. Compile with `adam` optimizer and `categorical_crossentropy` loss.
9. Use `accuracy` as the evaluation metric.
10. Train the model for **10 epochs**, using a `batch_size` of 128 and a **validation_split** 
of 0.1.
11. Plot training and validation accuracy/loss per epoch.
12. Evaluate the model on the test set and report the final test accuracy.
13. Use `model_predict()` to get predictions on test data.
14. Display a **confusion matrix** using `seaborn.heatmap`.
15. Print a **classification report** showing precision, recall, and F1-score.
16. Save the trained model in the **native Keras format**.

### Links 🔗

## 🛠️ My process

### Built with 🧱

### What I learned 🧠

### Continued development 🌱

### Useful resources 📚

## 👩🏽‍💻 Author

- LinkedIn - [Grace Sampao](https://www.linkedin.com/in/grace-sampao)
- GitHub - [@nadupoy](https://github.com/nadupoy)
- X - [@grace_sampao](https://x.com/grace_sampao)
- [Blog](https://nadupoy.github.io/)