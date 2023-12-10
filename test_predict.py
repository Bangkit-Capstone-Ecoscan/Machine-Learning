# Importing necessary libraries
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\Dunia Perkuliahan\Semester 5\Bangkit\Model\model_tf.h5')

# Define food categories
data_path = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset"
categories = sorted(os.listdir(os.path.join(data_path, "training")))  # Replace with your actual categories

def predict_food_size(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    prediction = model.predict(np.expand_dims(img, axis=0))
    food_category = categories[np.argmax(prediction)]

    # Estimate the size of the dish
    area = np.sum(img > 50)  # Thresholding to exclude background pixels
    if area < 10000:
        size = 'small'
    elif area < 30000:
        size = 'medium'
    else:
        size = 'large'

    return food_category, size

# Path to the folder containing test images
test_folder_path = 'D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Test Model'

# List of file paths in the test folder
file_paths = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path) if
              os.path.isfile(os.path.join(test_folder_path, file))]

# Test the model on each image in the folder
for i, file_path in enumerate(file_paths, 1):
    test_image_path = file_path
    food_category, size = predict_food_size(test_image_path)

    print(f"Item #{i}: {os.path.basename(file_path)}")
    print(f"Food Category: {food_category}")
    print(f"Estimated Size: {size}")
    print()
