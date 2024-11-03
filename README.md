# Handwritten Digit Recognition

This project implements a machine learning model to recognize handwritten digits using the MNIST dataset. The goal is to classify digits (0-9) from images of handwritten numbers with high accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
Handwritten digit recognition is a common task in computer vision, often used as a benchmark for image classification algorithms. In this project, a neural network is trained on the MNIST dataset to identify handwritten digits from 0 to 9.

**Model**: This project uses a convolutional neural network (CNN) to classify the digits. CNNs are effective in identifying patterns in images, making them ideal for handwritten digit recognition tasks.

## Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used in this project. It consists of:
- 60,000 training images
- 10,000 testing images
- Each image is a 28x28 grayscale image of a single handwritten digit

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Muskanverma24/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Install the required packages if you haven’t already.

3. Download the MNIST dataset:
   - The script will download it automatically if you are using a library like TensorFlow or PyTorch.
   - Alternatively, you can download it manually from [MNIST](http://yann.lecun.com/exdb/mnist/).

## Usage

For custom digit predictions, place your images in the `images/` folder and run

## Results
The model achieves an accuracy of around **98.73%** on the MNIST test set.

