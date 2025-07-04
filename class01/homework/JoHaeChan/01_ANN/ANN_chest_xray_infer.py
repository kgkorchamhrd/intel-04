import numpy as np
import matplotlib.pyplot as plt
import os
import random  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import glob

model = load_model("xray.h5")
test_dir = "./chest_xray/test"
test_n = test_dir+'NORMAL/'
test_p = test_dir+'PNEMONIA/'
FileListNormal = glob.glob(test_n+"*")
FileListPneumonia = glob.glob(test_p+"*")

normal_len = len(FileListNormal)
pneumonia_len = len(FileListPneumonia)

select_normal = np.random.randint(normal_len, size=20)
select_pneumonia = np.random.randint(pneumonia_len, size=20)

print(select_normal)
print(select_pneumonia)

data_test_sets = []
for index, position in enumerate(select_normal):
    data_test_sets.append((0, FileListNormal[position]))
for index, position in enumerate(select_pneumonia):
    data_test_sets.append((1, FileListPneumonia[position]))

exit()

categories = ["NORMAL", "PNEUMONIA"]

image_paths = []
for category in categories:
    category_path = os.path.join(test_dir, category)
    filenames = os.listdir(category_path)
    for filename in filenames:
        image_paths.append((os.path.join(category_path, filename), category))


random_images = random.sample(image_paths, 25)

plt.figure(figsize=(20, 20))

for idx, (img_path, label) in enumerate(random_images):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    prob = prediction[0][0]
    pred_class = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    plt.subplot(5, 5, idx + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"GT: {label}\nPred: {pred_class}\nPneumonia: {prob*100:.1f}%", fontsize=10)

plt.tight_layout()
plt.show()

