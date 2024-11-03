import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model


# loading model
model = load_model('model/digit_recognition_model.h5') 

img_path = r'recognition/images/image1.jpg' 
image_height, image_width = 28, 28 

# Reading the image using OpenCV
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize the image to the model's input size
img_resized = cv2.resize(img_gray, (image_width, image_height))
img_array = img_resized / 255.0

# Add channel dimension for grayscale
img_array = np.expand_dims(img_array, axis=-1)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)  

# Make a prediction
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction, axis=1)

# Display the image and prediction
plt.imshow(img_resized, cmap='gray')
plt.title(f'Predicted Class: {predicted_digit[0]}')
plt.show()