import os

data_path = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset"
categories = sorted(os.listdir(os.path.join(data_path, "training")))

print(categories)