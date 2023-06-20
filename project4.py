#%%
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
#%%
# 在开启对话session前，先创建一个 tf.ConfigProto() 实例对象

gpuConfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth=True



# 把你的配置部署到session  变量名 sess 无所谓
sess1 =tf.compat.v1.Session(config=gpuConfig)

#%% input training data
import tensorflow_datasets as tfds

dataset = tfds.load("stl10", split=tfds.Split.TRAIN,as_supervised=True)
dataset_test = tfds.load("stl10", split=tfds.Split.TEST,as_supervised=True)
#dataset_unlabel=tfds.load("stl10", split='unlabelled',as_supervised=True)
#images = np.asarray(list(dataset_unlabel.map(lambda x, y: x)))
dataset=np.asarray(list(dataset.map(lambda x, y: x)))
dataset_test=np.asarray(list(dataset_test.map(lambda x, y: x)))
x_train,y_train=train_test_split(np.asarray(np.concatenate((dataset, dataset_test), axis=0)),test_size=0.1)


x_train=x_train/255


y_train=y_train/255
#for i in range(len(y_train)):
#    y_train[i]=y_train[i]/255

#%%
import matplotlib.pyplot as plt
i=0
fig = plt.figure(figsize=(10, 7))
rows = 5
columns = 2
for im in x_train:
    i+=1
    if i<10:
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)
    else:
        StopIteration
#%%
DATASET_SIZE=len(dataset_unlabel)
train_size = int(0.9 * DATASET_SIZE)
val_size = int(0.1 * DATASET_SIZE)
#dataset_unlabel_2=tf.gather(dataset_unlabel,[0])
dataset_unlabel= dataset_unlabel.shuffle(buffer_size=1000,reshuffle_each_iteration=False)

train_dataset = dataset_unlabel.take(train_size)
val_dataset = dataset_unlabel.skip(train_size)
#%% show picture
#vis = tfds.visualization.show_examples(val_dataset, ds_info=False)

it = iter(val_dataset)
#print(next(it)[0].numpy)
images, labels = next(it)
import matplotlib.pyplot as plt
#plt.imshow(images)
#print(labels)
i=0
fig = plt.figure(figsize=(10, 7))
rows = 5
columns = 2
for images,labels in val_dataset:
    i+=1
    if i<10:
        fig.add_subplot(rows, columns, i)
        plt.imshow(images)
    else:
        StopIteration
    

#    print(labels)
#%%  efficientnet_stage4_16layer
n=100
inputshape=(96,96,3)
def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])

model_input = Input(shape=inputshape)
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(n,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,n,n,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,n*6,n,"stage1_block2")
#stage1_block2=efficeint_block_same_channel(stage1_block2,n*6,n,"stage1_block3")
#stage1_block2=efficeint_block_same_channel(stage1_block2,n*6,n,"stage1_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,n*6*2,n*2,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,n*6*2,n*2,"stage2_block2")
#stage2_block3=efficeint_block_same_channel(stage2_block2,n*6*2,n*2,"stage2_block3")
#stage2_block4=efficeint_block_same_channel(stage2_block3,n*6*2,n*2,"stage2_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block2)


stage3_block1=efficeint_block_different_channel(pool1,n*6*4,n*4,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,n*6*4,n*4,"stage3_block2")
#stage3_block3=efficeint_block_same_channel(stage3_block2,n*6*4,n*4,"stage3_block3")
#stage3_block4=efficeint_block_same_channel(stage3_block3,n*6*4,n*4,"stage3_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block2)


stage4_block1=efficeint_block_different_channel(pool1,n*6*8,n*8,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,n*6*8,n*8,"stage4_block2")
#stage4_block2=efficeint_block_same_channel(stage4_block2,n*6*8,n*8,"stage4_block3")
#stage4_block2=efficeint_block_same_channel(stage4_block2,n*6*8,n*8,"stage4_block4")
#pool1=tf.keras.layers.MaxPooling2D(
#    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
#  )(stage4_block2)

#flatten1= tf.keras.layers.Flatten()(pool1)

#dense1=tf.keras.layers.Dense(1024, activation='swish',kernel_initializer='random_normal'
#            )(flatten1)
#dense1=tf.keras.layers.Dropout(0.2)(dense1)
#dense1=tf.keras.layers.Dense(256, activation='swish',kernel_initializer='random_normal'
 #           )(dense1)
#dense1=tf.keras.layers.Dropout(0.2)(dense1)
#dense1=tf.keras.layers.Dense(64, activation='swish',kernel_initializer='random_normal'
#            )(dense1)
#dense1=tf.keras.layers.Dropout(0.2)(dense1)
#dense3=tf.keras.layers.Dense(outputclass, activation='softmax',kernel_initializer='random_normal'
# )(dense1)



conv1=tf.keras.layers.Conv2D(10,(3,3), padding="same")(stage4_block2)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_different_channel(relu,n*8,n*8,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,n*6*8,n*8,"stage1_block2")
#stage1_block2=efficeint_block_same_channel(stage1_block2,n*6,n,"stage1_block3")
#stage1_block2=efficeint_block_same_channel(stage1_block2,n*6,n,"stage1_block4")
pool1=tf.keras.layers.UpSampling2D(
    size=(2, 2), data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,n*6*4,n*4,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,n*6*4,n*4,"stage2_block2")
#stage2_block3=efficeint_block_same_channel(stage2_block2,n*6*4,n*4,"stage2_block3")
#stage2_block4=efficeint_block_same_channel(stage2_block3,n*6*4,n*4,"stage2_block4")
pool1=tf.keras.layers.UpSampling2D(
    size=(2, 2), data_format=None,
  )(stage2_block2)


stage3_block1=efficeint_block_different_channel(pool1,n*6*2,n*2,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,n*6*2,n*2,"stage3_block2")
#stage3_block3=efficeint_block_same_channel(stage3_block2,n*6*2,n*2,"stage3_block3")
#stage3_block4=efficeint_block_same_channel(stage3_block3,n*6*2,n*2,"stage3_block4")
pool1=tf.keras.layers.UpSampling2D(
    size=(2, 2), data_format=None,
  )(stage3_block2)


stage4_block1=efficeint_block_different_channel(pool1,n*6,n,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,n*6,n,"stage4_block2")
conv1=tf.keras.layers.Conv2D(3,(3,3), padding="same")(stage4_block2)
#batch_norm=tf.keras.layers.BatchNormalization()(conv1)
sigmoid=tf.keras.activations.sigmoid(conv1)

decoder= Model(inputs=model_input, outputs=sigmoid)
loss_fn=tf.keras.losses.MeanSquaredError()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    
    
    )
#opt = mixed_precision.LossScaleOptimizer(opt )

plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

decoder.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(decoder.summary())
# %%
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 5, factor= 0.5, min_lr= 1e-10, verbose=1)
tensorboard=tf.keras.callbacks.TensorBoard(log_dir="C:/Users/ryan7/Documents/GitHub/ai_project4/project4_predict_test_logs")
checkpoint_filepath = 'C:/Users/ryan7/Documents/GitHub/ai_project4/checkpoint'
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True)]
datagen = ImageDataGenerator()

decoder.fit(x=datagen.flow(x_train, x_train,batch_size=5),
  validation_data = (y_train,y_train),epochs=50,verbose=1,#batch_size=10,

  shuffle=True, use_multiprocessing=False#,workers=6
  ,callbacks = [early_stop, rlrop,tensorboard,model_checkpoint_callback])
# %%
# %%
import matplotlib.pyplot as plt
plt.plot(decoder.history.history['accuracy'])
plt.plot(decoder.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%
selfpredict=decoder.predict(y_train[[0]])
plt.imshow(selfpredict[0].astype('float32'))
# %%
decoder.save("C:/Users/ryan7/Documents/GitHub/ai_project4/decoder_efficient_output3")
#%%
weights=decoder.get_weights()
# %% add_7 is encoder end

encoder= Model(inputs=decoder.input,outputs=decoder.get_layer("add_7").output)
plot_model(encoder, to_file='model_plot_encoder.png', show_shapes=True, show_layer_names=True)


# %%
weights_encoder=encoder.get_weights()
# %%
