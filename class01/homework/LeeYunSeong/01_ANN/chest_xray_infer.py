import numpy as np # forlinear algebra
import matplotlib.pyplot as plt
import cv2
import copy

model = keras.models.load_model("./medical_ann.h5")
test_folder = '/home/workspace/chest_xray/test/'
train_normal_filename = test_folder+'NORMAL/'
train_pneumonia_filename = test_folder+'PNEUMONIA/'
FileListNormal = glob.glob(test_normal_filename+"*")
FileListPneumonia = glob.glob(test_Pneumonia_filename+"*")

normal_len = len(FileListNormal)
pneumonia_len = len(FileListPneumonia)

select_normal = np.random.randint(normal_len, size=20)
select_pneumonia = np.random.randint(pneumonia_len, size=20)

print(select_normal)
print(select_pneumonia)

data_test_sets = []
for index, position in enumerate(select_normal):
    data_test_sets.append((0, FileListNormal[position]))
for index, position in enumerate(select_pneumonianormal):
    data_test_sets.append((1, FileListPneumonia[position]))


