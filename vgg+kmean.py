#%%
# for loading/processing the images  
#from keras.preprocessing.image import load_img 
#from keras.preprocessing.image import img_to_array 
#from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
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
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
model = VGG16(weights="imagenet", include_top=False,input_tensor= Input(shape=(96, 96, 3)))
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.output)
plot_model(model, to_file='model_plot_model.png', show_shapes=True, show_layer_names=True)
#%%


x = Flatten()(model.output)
encoded = Dense(10, activation='softmax', name='encoded')(x)
model=Model(inputs=model.input,outputs=encoded)
plot_model(model, to_file='model_plot_encoder.png', show_shapes=True, show_layer_names=True)
from tensorflow.keras.optimizers import Adam
opt=Adam(
    learning_rate=0.0001,
    
    
    )
loss = 'sparse_categorical_crossentropy'
#opt = mixed_precision.LossScaleOptimizer(opt )

#plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy'])
print(model.summary())
#%%
#%%
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 5, factor= 0.5, min_lr= 1e-10, verbose=1)
tensorboard=tf.keras.callbacks.TensorBoard(log_dir="C:/Users/ryan7/Documents/GitHub/ai_project4/project4_predict_test_logs")
checkpoint_filepath = 'C:/Users/ryan7/Documents/GitHub/ai_project4/checkpoint'
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True)]
datagen = ImageDataGenerator()
#x=le.fit_transform(x_label.reshape(1, -1))
model.fit(x=datagen.flow(x_train, x_label,batch_size=5),
  validation_data = (y_train,y_label),epochs=50,verbose=1,#batch_size=10,

  shuffle=True, use_multiprocessing=False#,workers=6
  ,callbacks = [early_stop, rlrop,tensorboard,model_checkpoint_callback])
#%%
model.save("C:/Users/ryan7/Documents/GitHub/ai_project4/vgg16")
#%%
features = model.predict(y_train)
print(features.shape)
y_reduce_pattern=features.reshape(1000,3*3*512)
#pca = PCA(n_components=100, random_state=22)
#pca.fit(y_reduce_pattern)
#y_reduce_pattern = pca.transform(y_reduce_pattern)

#y_reduce_pattern = manifold.TSNE(n_components=3, init='random', random_state=5, verbose=1).fit_transform(y_reduce_pattern)
#%%
n_clusters=10
kmeans = KMeans(n_clusters=n_clusters, init="random")
y_pred_kmeans = kmeans.fit_predict(y_reduce_pattern)
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred_kmeans)

#%%

from sklearn_extra.cluster import KMedoids
n_clusters=10
kmeans = KMedoids(n_clusters=n_clusters, n_init="auto")
y_pred_kmeans = kmeans.fit_predict(y_reduce_pattern)
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred_kmeans)
# %%
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
#%%
a=0
for i in range(len(confusion_matrix)):
    a+=max(confusion_matrix[:,i])
print(a)
# %%
from sklearn.metrics.cluster import completeness_score
print (completeness_score(y_label, y_pred_kmeans))

# %%
i=0
fig = plt.figure(figsize=(100, 70))
rows = 1
columns = 10
for im in y_train:
    i+=1
    if i<10:
        fig.add_subplot(rows,columns,i)
        plt.imshow(im)
        print(y_label[i])
        print(y_pred_kmeans[i])
    else:
        StopIteration
# %%

plt.imshow(color.rgb2gray(x_train[1]))
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
from sklearn.metrics.cluster import completeness_score
print (completeness_score(y_label, clusters))
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, clusters)
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
# %%
clustering = DBSCAN(eps=3, min_samples=2)
clustering.fit(y_reduce_pattern)