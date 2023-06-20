#%%
# for loading/processing the images  
#from keras.preprocessing.image import load_img 
#from keras.preprocessing.image import img_to_array 
#from keras.applications.vgg16 import preprocess_input 

# models 
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
from skimage import color
from skimage import io
from sklearn import manifold, datasets
from sklearn.cluster import DBSCAN
from rembg import remove
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

import cv2
#%%
model = InceptionResNetV2(weights="imagenet", include_top=False,input_tensor= Input(shape=(96, 96, 3)))
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.output)
plot_model(model, to_file='model_plot_model.png', show_shapes=True, show_layer_names=True)
#%% load img test

data_path = "/mnt/c/Users/ryan7/Downloads/410873001"

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
#%%
#%%
features = model.predict(base_x_test/255)
print(features.shape)
y_reduce_pattern=features.reshape(1000,1*1*1536)
#pca = PCA(n_components=100, random_state=22)
#pca.fit(y_reduce_pattern)
#y_reduce_pattern = pca.transform(y_reduce_pattern)

#y_reduce_pattern = manifold.TSNE(n_components=3, init='random', random_state=5, verbose=1).fit_transform(y_reduce_pattern)
#%%
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

dis=sch.linkage(y_reduce_pattern,metric='euclidean',method='ward')
sch.dendrogram(dis)
plt.title('Hierarchical Clustering')
plt.show()
k=10
clusters=sch.fcluster(dis,k,criterion='maxclust')
#%%
path_to_file = "/mnt/c/Users/ryan7/Downloads/"
with open(path_to_file + "410873001.txt", "w") as g:
    for t in range(1,testnumber+1):
        for j in range(testnumber):
            if int(FileName_test[j])==t:
            
                g.write(FileName_test[j] + " " + str(int(clusters[j])) + "\n")
