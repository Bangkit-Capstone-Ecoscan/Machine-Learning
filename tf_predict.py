import os
import PIL
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd


# Assigning label names to the corresponding indexes
labels = {
    0: 'Bread',
    1: 'Dairy product',
    2: 'Dessert',
    3: 'Egg',
    4: 'Fried food',
    5: 'Meat',
    6: 'Noodles-Pasta',
    7: 'Rice',
    8: 'Seafood',
    9: 'Soup',
    10: 'Vegetable-Fruit'
}

#load model
model = load_model('D:\Dunia Perkuliahan\Semester 5\Bangkit\Model\model_tf.h5')

# Loading images and their predictions
path_eval = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset\evaluation"

# code to open the image

img = PIL.Image.open("D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Test Model\\martabak.jpeg")


img = PIL.Image.open("D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Test Model\\mi_kuah.jpg")

# resizing the image to (256,256)
img = img.resize((256, 256))
# converting image to array
img = np.asarray(img, dtype=np.float32)
# normalizing the image
img = img / 255
# reshaping the image in to a 4D array
img = img.reshape(-1, 256, 256, 3)
# making prediction of the model
predict = model.predict(img)
# getting the index corresponding to the highest value in the prediction
predict = np.argmax(predict)


# Read data from csv
df = pd.read_csv('Carbon_Emission_of_Foods.csv')
df2 = pd.read_csv('food_nutrition.csv')

# Get the predicted class from the model
pred = labels[predict]

# Filter the DataFrame based on the predicted class
filter_carbon = df[df['id'] == pred.lower()]
filter_nutrition = df2[df2['name'] == pred.lower()]

# Extract the 'value' for the predicted class
carbon = filter_carbon['value'].values[0]
protein = filter_nutrition['protein'].values[0]
calcium = filter_nutrition['calcium'].values[0]
fat = filter_nutrition['fat'].values[0]
carbohydrates = filter_nutrition['carbohydrates'].values[0]
vitamins = filter_nutrition['vitamins'].values[0]

#print output
print("Prediction: " + pred)
print(f'Carbon Emission: {carbon:.2f} kg CO2')
print(f'Protein: {protein:.2f} g')
print(f'Calcium: {calcium:.2f} mg')
print(f'Fat: {fat:.2f} g')
print(f'Carbohydrates: {carbohydrates:.2f} g')
print(f'Vitamins: {vitamins}')