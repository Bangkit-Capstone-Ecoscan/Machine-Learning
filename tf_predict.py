import os
import PIL
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


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

model = load_model('D:\Dunia Perkuliahan\Semester 5\Bangkit\Model\model_tf.h5')

# Loading images and their predictions
prediction = []
original = []
image = []
count = 0
path_eval = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset\evaluation"

# for i in os.listdir(path_eval):
#     for item in os.listdir(os.path.join(path_eval, i)):

# code to open the image
img = PIL.Image.open("D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Test Model\\sego.jpg")
# resizing the image to (256,256)
img = img.resize((256, 256))
# appending image to the image list
image.append(img)
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
# appending the predicted class to the list
prediction.append(labels[predict])
# appending original class to the list
original.append("Rice")

# Visualizing the results
# fig=plt.figure(figsize = (100,100))

print("Prediction: " + prediction[0])
# plt.xlabel("Prediction -" + prediction[0] +"   Original -" + original[0])
# plt.imshow(image[0])
# fig.tight_layout()
# plt.show()