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
#y_train_gray=color.rgb2gray(y_train)
#y_train_gray=np.asarray([y_train_gray,y_train_gray,y_train_gray])
#y_train_gray=np.transpose(y_train_gray, (1, 2, 3, 0))

x_train=x_train/255



#%%
# load model
model = InceptionResNetV2(weights="imagenet", include_top=False,input_tensor= Input(shape=(96, 96, 3)))
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.output)
plot_model(model, to_file='model_plot_model.png', show_shapes=True, show_layer_names=True)
#%%
features = model.predict(y_train)
print(features.shape)
y_reduce_pattern=features.reshape(1000,1*1*1536)
#pca = PCA(n_components=100, random_state=22)
#pca.fit(y_reduce_pattern)
#y_reduce_pattern = pca.transform(y_reduce_pattern)

#y_reduce_pattern = manifold.TSNE(n_components=3, init='random', random_state=5, verbose=1).fit_transform(y_reduce_pattern)
#%%
n_clusters=10
kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
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
a=0
for i in range(len(confusion_matrix)):
    a+=max(confusion_matrix[:,i])
print(a)


# %%
fig = plt.figure(figsize=(100, 70))
rows = 1
columns = 10
for i in range(10):
    fig.add_subplot(rows,columns,i+1)
    plt.imshow(y_train[y_pred_kmeans==6][i])
# %%
x= remove(y_train[1])
#plt.imshow(x)

plt.imshow(rgba2rgb(x))
# %%
from sklearn.cluster import kmeans_plusplus
kmplus=kmeans_plusplus(n_clusters=n_clusters)
y_pred_kmeansplus=kmplus.fit_predict(y_reduce_pattern)
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred_kmeansplus)
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
# %%
#%%
#weights_encoder=encoder.get_weights()
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

dis=sch.linkage(y_reduce_pattern,metric='euclidean',method='ward')
sch.dendrogram(dis)
plt.title('Hierarchical Clustering')
plt.show()
k=10
clusters=sch.fcluster(dis,k,criterion='maxclust')
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, clusters-1)
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
# %%
# %%
fig = plt.figure(figsize=(100, 70))
rows = 10
columns = 10
for i in range(1,11):
    for j in range(1,11):
        fig.add_subplot(rows,columns,(j-1)*10+i)
        plt.imshow(y_train[clusters==j][i])


#%%
pca = PCA(n_components=2, random_state=22)
pca.fit(y_reduce_pattern)
y_reduce_pattern = pca.transform(y_reduce_pattern)
#%%
color=["#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF",
       "#FF00FF","#FF0088","#FF8800","#00FF99","#7700FF"]
plt.rcParams["figure.figsize"] = (18,9)
fig , ax = plt.subplots()
plt.subplot(1, 2, 1)
for i in range(0,10):
    BOOL=(y_pred_kmeans==i)
    plt.scatter(y_reduce_pattern[BOOL,0],y_reduce_pattern[BOOL,1],c=color[i],label=i)
plt.title("k-mean clustering",fontsize=30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.subplot(1, 2, 2)
for i in range(0,10):
    BOOL=(y_label==i)
    plt.scatter(y_reduce_pattern[BOOL,0],y_reduce_pattern[BOOL,1],c=color[i],label=i)
plt.title("original data",fontsize=30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.show()
# %%
pca = PCA(n_components=2, random_state=22)
pca.fit(y_train.reshape(1000,96*96*3))
y_reduce_pattern = pca.transform(x_train.reshape(12000,96*96*3))
color=["#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF",
       "#FF00FF","#FF0088","#FF8800","#00FF99","#7700FF"]
plt.rcParams["figure.figsize"] = (18,9)
fig , ax = plt.subplots()
plt.subplot(1, 2, 1)
for i in range(0,10):
    BOOL=(y_pred_kmeans==i)
    plt.scatter(y_reduce_pattern[BOOL,0],y_reduce_pattern[BOOL,1],c=color[i],label=i)
plt.title("k-mean clustering",fontsize=30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.subplot(1, 2, 2)
for i in range(0,10):
    BOOL=(x_label==i)
    plt.scatter(y_reduce_pattern[BOOL,0],y_reduce_pattern[BOOL,1],c=color[i],label=i)
plt.title("original data",fontsize=30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.show()
# %%
pca = PCA(n_components=2, random_state=22)
pca.fit(y_train.reshape(1000,96*96*3))
y_reduce_pattern = pca.transform(y_train.reshape(1000,96*96*3))
color=["#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF",
       "#FF00FF","#FF0088","#FF8800","#00FF99","#7700FF"]
plt.rcParams["figure.figsize"] = (18,9)
fig , ax = plt.subplots()
plt.subplot(1, 2, 1)
for i in range(0,10):
    BOOL=(clusters==i)
    plt.scatter(y_reduce_pattern[BOOL,0],y_reduce_pattern[BOOL,1],c=color[i],label=i)
plt.title("k-mean clustering",fontsize=30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.subplot(1, 2, 2)
for i in range(0,10):
    BOOL=(y_label==i)
    plt.scatter(y_reduce_pattern[BOOL,0],y_reduce_pattern[BOOL,1],c=color[i],label=i)
plt.title("original data",fontsize=30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.show()
# %%
def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

for i in range(len(y_train)):
    y_train[i]= rgba2rgb(remove(y_train[i]))