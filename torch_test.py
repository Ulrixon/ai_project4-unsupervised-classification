#%%
import torch
torch.cuda.is_available()

# %%
import numpy as np
#x=np.load('/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/SPICE/results/stl10/embedding/feas_moco_512_l2.npy')
# %%
from PIL import Image
import struct
import matplotlib.pyplot as plt
size = 5, 5
arr = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
data = struct.pack('B'*len(arr), *[pixel*255 for pixel in arr])
img = Image.frombuffer('L', size, data)
plt.imshow(img)
# %%
d = np.load("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/SPICE/results/stl10/eval/proto/test_gt.npy")
e = np.load("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/SPICE/results/stl10/eval/proto/test_label.npy")
# %%
import sklearn.metrics
import seaborn as sns
confusion_matrix = sklearn.metrics.confusion_matrix(d, e)
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
# %%
import numpy as np
from matplotlib import pylab as plt
n,w, h,rgb = 8000,96,3, 96
with open("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/SPICE/datasets/stl10/stl10_binary/test_X.bin", mode='rb') as f:
    A = np.fromfile(f, dtype=np.uint8,count=n*w*h*rgb).reshape(n,h,w,rgb)
plt.imshow(np.transpose(A[7999], (2, 1, 0)))
plt.imshow(np.transpose(A[7998], (2, 1, 0)))
plt.imshow(np.transpose(A[7997], (2, 1, 0)))
# %%
import tensorflow_datasets as tfds
dataset_test_tfds = tfds.load("stl10", split=tfds.Split.TEST,as_supervised=True)
y_train=np.asarray(list(dataset_test_tfds.map(lambda x, y: x)))
y_train=np.transpose(y_train,(0,3,2,1))
#plt.imshow(y_train[7999])
y_train.astype('uint8').tofile("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/test.bin")

# %%
n,w, h,rgb = 8000,96,3, 96
with open("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/test.bin", mode='rb') as f:
    A = np.fromfile(f, dtype=np.uint8,count=n*w*h*rgb).reshape(n,h,w,rgb)

# %%










#%% load img test
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
data_path = "/mnt/c/Users/ryan7/Downloads/Test_data"

base_x_test = []

labelPath = data_path + r"/"

FileName_test = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]

for j in range(len(FileName_test)):
    path = labelPath + r"/" + FileName_test[j]

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    base_x_test.append(img)


testnumber = len(base_x_test)
base_x_test=np.array(base_x_test)



for j in range(0,len(FileName_test)):
    FileName_test[j]=FileName_test[j].removesuffix(".png")


base_x_test=np.transpose(base_x_test,(0,3,2,1))
#plt.imshow(y_train[7999])
base_x_test.astype('uint8').tofile("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/test_X.bin")
fake_test_y=np.zeros(testnumber)
fake_test_y.astype('uint8').tofile("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/test_y.bin")
# %%
n,w, h,rgb = 2490,96,3, 96
with open("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/test.bin", mode='rb') as f:
    A = np.fromfile(f, dtype=np.uint8,count=n*w*h*rgb).reshape(n,h,w,rgb)
#%%
y_pred = np.load("/mnt/c/Users/ryan7/Documents/GitHub/ai_project4/SPICE/results/stl10/eval/proto/test_label.npy")
path_to_file = "/mnt/c/Users/ryan7/Downloads/"
with open(path_to_file + "410873001.txt", "w") as g:
    #for t in range(1,testnumber+1):
    for j in range(testnumber):
        #f int(FileName_test[j])==t:
            
        g.write(FileName_test[j] + " " + str(int(y_pred[j])) + "\n")

# %%
