from tensorflow import keras
import glob
import numpy as np

model = keras.models.load_model("./medical_ann.h5")
test_folder = './chest_xray/chest_xray/test/'
test_normal_filename = test_folder+'NORMAL/'
test_pneumonia_filename = test_folder+'PNEUMONIA/'
FileListNormal = glob.glob(test_normal_filename+"*")
FileListPneumonia = glob.glob(test_pneumonia_filename+"*")

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
    data_test_sets.append((0, FileListPneumonia[position]))

print(len(data_test_sets))
