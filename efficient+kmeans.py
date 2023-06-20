#%%
# for loading/processing the images  
#from keras.preprocessing.image import load_img 
#from keras.preprocessing.image import img_to_array 
#from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.efficientnet import EfficientNetB7
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
#%%
#%% input training data
import tensorflow_datasets as tfds

dataset_tfds = tfds.load("stl10", split=tfds.Split.TRAIN,as_supervised=True)
dataset_test_tfds = tfds.load("stl10", split=tfds.Split.TEST,as_supervised=True)
dataset_tfds=dataset_tfds.concatenate(dataset_test_tfds)
test_dataset =dataset_tfds.take(1000) 
train_dataset = dataset_tfds.skip(1000)
#dataset_unlabel=tfds.load("stl10", split='unlabelled',as_supervised=True)
#images = np.asarray(list(dataset_unlabel.map(lambda x, y: x)))
x_train=np.asarray(list(train_dataset.map(lambda x, y: x)))
x_label=np.asarray(list(train_dataset.map(lambda x, y: y)))
y_train=np.asarray(list(test_dataset.map(lambda x, y: x)))
y_label=np.asarray(list(test_dataset.map(lambda x, y: y)))
y_train=y_train/255
y_train_gray=color.rgb2gray(y_train)
y_train_gray=np.asarray([y_train_gray,y_train_gray,y_train_gray])
y_train_gray=np.transpose(y_train_gray, (1, 2, 3, 0))

x_train=x_train/255



#%%
# load model
model = EfficientNetB7(weights="imagenet", include_top=True,input_tensor= Input(shape=(96, 96, 3)))
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
plot_model(model, to_file='model_plot_model.png', show_shapes=True, show_layer_names=True)

features = model.predict(y_train)
print(features.shape)
y_reduce_pattern=features.reshape(1000,3*3*2560)
pca = PCA(n_components=100, random_state=22)
pca.fit(y_reduce_pattern)
y_reduce_pattern = pca.transform(y_reduce_pattern)

#%%
n_clusters=10
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred_kmeans = kmeans.fit_predict(y_reduce_pattern)
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred_kmeans)
# %%
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
# %%
from sklearn.metrics.cluster import completeness_score
print (completeness_score(y_label, y_pred_kmeans))
# %%
