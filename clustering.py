# %%
import os
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from pandas import array
import sklearn
import tensorflow as tf
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from PIL import Image
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
#from gradient_accumulator.GAModelWrapper import GAModelWrapper
from tensorflow.keras import mixed_precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
import numpy as np
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



x_train=x_train/255


y_train=y_train/255
#%%
# load model
model = VGG16(weights="imagenet", include_top=False,input_tensor= Input(shape=(96, 96, 3)))
# remove the output layer
shape_before_flattening = K.int_shape(model.output)
#x = Flatten()(model.output)
#encoded = Dense(10, activation='relu', name='encoded')(x)
#x = Dense(np.prod(shape_before_flattening[1:]),
#                activation='relu')(encoded)
#x = Reshape(shape_before_flattening[1:])(x)
x= Conv2D(512, (3, 3), activation='relu', padding='same')(model.layers[-2].output)
x= Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x= Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x= Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x= Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x= Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x= Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x= Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x= Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x= Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x= Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x= Conv2D(64, (3, 3), activation='relu', padding='same')(x)
decoded= Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder=Model(inputs=model.input, outputs=decoded, name='AE')

loss_fn=tf.keras.losses.MeanSquaredError()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    
    
    )
#opt = mixed_precision.LossScaleOptimizer(opt )

plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

autoencoder.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(autoencoder.summary())

# %%
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 5, factor= 0.5, min_lr= 1e-10, verbose=1)
tensorboard=tf.keras.callbacks.TensorBoard(log_dir="C:/Users/ryan7/Documents/GitHub/ai_project4/project4_predict_test_logs")
checkpoint_filepath = 'C:/Users/ryan7/Documents/GitHub/ai_project4/checkpoint'
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True)]
datagen = ImageDataGenerator()

autoencoder.fit(x=datagen.flow(x_train, x_train,batch_size=5),
  validation_data = (y_train,y_train),epochs=50,verbose=1,#batch_size=10,

  shuffle=True, use_multiprocessing=False#,workers=6
  ,callbacks = [early_stop, rlrop,tensorboard,model_checkpoint_callback])
#%%
b=autoencoder.predict(y_train[0].reshape(1,96,96,3))
plt.imshow(b[0].astype('float32'))
plt.imshow(y_train[0])


#%%
encoder = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer("encoded").output)
plot_model(encoder, to_file='model_plot_model.png', show_shapes=True, show_layer_names=True)

#%%
n_clusters=10
batch_size=5
x=y_train
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim,), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(tf.keras.layers.Flatten()(encoder.output))
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer='adam', loss='kld')
#%%
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(y_train))
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred)
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
#%%
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
#%%
# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])
tol = 0.001 # tolerance threshold to stop training


#%%
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


#%%

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y_label is not None:
            acc = np.round(acc(y_label, y_pred), 5)
            nmi = np.round(nmi(y_label, y_pred), 5)
            ari = np.round(ari(y_label, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
#%%
q = model.predict(x, verbose=0)
p = target_distribution(q)
# evaluate the clustering performance
y_pred = q.argmax(1)
if y_label is not None:
    acc = np.round(acc(y_label, y_pred), 5)
    nmi = np.round(nmi(y_label, y_pred), 5)
    ari = np.round(ari(y_label, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()