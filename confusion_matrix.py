import os
import numpy as np
import PIL
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model('D:\Dunia Perkuliahan\Semester 5\Bangkit\Model\model_tf.h5')

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

# Specify the path to the evaluation data
path_eval = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset\evaluation"

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Loop through the images in the evaluation directory
for label_folder in os.listdir(path_eval):
    label_path = os.path.join(path_eval, label_folder)

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        # Open and preprocess the image
        img = PIL.Image.open(img_path)
        img = img.resize((256, 256))
        img_array = np.asarray(img, dtype=np.float32) / 255
        img_array = img_array.reshape(-1, 256, 256, 3)

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        # Append true and predicted labels to the lists
        true_labels.append(label_folder)
        predicted_labels.append(predicted_label)

# Convert string labels to integer labels using LabelEncoder
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Generate and display the confusion matrix
cm = confusion_matrix(true_labels_encoded, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels.values(), yticklabels=labels.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Display the classification report
print('Classification Report:\n',
      classification_report(true_labels_encoded, predicted_labels, target_names=labels.values()))
