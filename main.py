# NOTE: Comment or Uncomment according to the requirements of SE(squeezed excitation)
from google.colab import drive
drive.mount('/content/drive')
data_path = "/content/drive/MyDrive/lady'sfinger_100/"
import pandas as pd
import os

image_list = []
label_list = []
label_int =[]

for i, folder in enumerate(os.listdir(data_path)):
  images = os.listdir(f"{data_path}/" + folder + "/")
  for row in range(len(images)):
    image_list.append(f"{data_path}/" + folder + "/" + images[i])
    label_list.append(folder)
    label_int.append(i)
{i:x for i, x in enumerate(os.listdir(data_path))}
sub = pd.DataFrame(image_list)
sub.columns = ["imagename"]
sub["label_int"] = label_int
sub["label_list"] = label_list

sub.to_csv("dataset.csv")
sub.label_int.unique()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
X = sub["imagename"].values
y = sub["label_int"].values
skf.get_n_splits(X, y)
for train_index, test_index in skf.split(X,y):
     X_train, X_test = sub.iloc[train_index], sub.iloc[test_index]
     y_train, y_test = y[train_index], y[test_index]
X.shape
X_train.shape , X_test.shape
IMG_HEIGHT = 256
IMG_WIDTH = 384
IMG_CHANNELS = 1
IMG_COUNT = 10
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import classification_report
import tensorflow as tf

def BatchActivate(x):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('elu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1,1), padding='same', activation=True):
    x = tf.keras.layers.Conv3D(filters, size, strides=strides, padding=padding,kernel_initializer='Orthogonal')(x)
    if activation == True:
        x = BatchActivate(x)
    return x


def residual_block_dual(blockInput, num_filters=16, batch_activate = False):

    x_side = convolution_block(blockInput, num_filters,(3,3,3))

    x = BatchActivate(blockInput)
    x1 = convolution_block(x, num_filters, (3,3,3) ,activation=False)

    x2 = convolution_block(x1, num_filters, (3,3,3), activation=False)
    x2 = BatchActivate(x2)
    x2_add = tf.keras.layers.Add()([x1,x2])
    x3 = convolution_block(x2_add, num_filters, (5,5,5), activation=False)
    x3 = tf.keras.layers.Add()([x3,x_side])
    x4 = convolution_block(x3, num_filters, (3,3,3), activation=False)

#    x = Add()([Squeeze_excitation_layer(x4),x_side])
    if batch_activate:
        x4 = BatchActivate(x4)
    return x4

def residual_block(blockInput, num_filters=16, batch_activate = False):

    x_side = convolution_block(blockInput, num_filters,(3,3,3))

    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3,3) ,activation=True)
#    x = PReLU(shared_axes=[1, 2])(x)

    x= convolution_block(x, num_filters, (3,3,3), activation=True)

#    x = convolution_block(x, num_filters, (3,3), activation=True)

#    x = BatchActivate(x)
    x=Squeeze_excitation_layer_3D(x)
    x = tf.keras.layers.Add()([x,x_side])
    if batch_activate:
        x = BatchActivate(x)
    return x

import numpy as np

def Squeeze_excitation_layer_3D(input_x):
    ratio = 4
    out_dim =  int(np.shape(input_x)[-1])
    squeeze = tf.keras.layers.GlobalAveragePooling3D()(input_x)
    excitation = tf.keras.layers.Dense(units=int(out_dim / ratio))(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=out_dim)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([-1,1,1,out_dim])(excitation)
    scale = tf.keras.layers.multiply([input_x, excitation])

    return scale



def resnet3DClassifier(input_shape=(IMG_COUNT,IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS),learningRate=0.001,use_se_module=True):

    dropout_keep_prob =0.1
    '''

    Network with multiple attention

    '''
    start_neurons=32
    DropoutRatio = 0.2
    inputs = tf.keras.layers.Input(input_shape)

#    coord0=CoordinateChannel2D()(inputs)
    # 101 -> 50
    conv1 = tf.keras.layers.Conv3D(start_neurons, (3,3,3), activation='elu', padding="same")(inputs)
    conv1 = residual_block(conv1,start_neurons,True)
    conv1 = residual_block(conv1,start_neurons, True)
#    conv1=csse_block(conv1,'prefix_conv_1')
#    conv1 = nonLocalAttention(conv1)
    pool1 = tf.keras.layers.MaxPooling3D((2, 2,2))(conv1)
    pool1 = tf.keras.layers.Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = tf.keras.layers.Conv3D(start_neurons * 1, (3,3,3), activation='elu', padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 1,True)
    conv2 = residual_block(conv2,start_neurons * 1, True)

    pool2 = tf.keras.layers.MaxPooling3D((2, 2,2))(conv2)
    pool2 = tf.keras.layers.Dropout(DropoutRatio)(pool2)

#    # 25 -> 12
    conv3 = tf.keras.layers.Conv3D(start_neurons * 2, (3,3,3), activation='elu', padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 2)
    conv3 = residual_block(conv3,start_neurons * 2, True)


    pool3 = tf.keras.layers.MaxPooling3D((2,2,2))(conv3)
    pool3 = tf.keras.layers.Dropout(DropoutRatio)(pool3)

#    # 12 -> 6
    conv4 = tf.keras.layers.Conv3D(start_neurons * 3, (3, 3,3), activation='elu', padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 3)
    conv4 = residual_block(conv4,start_neurons * 3, True)


    x1 = tf.keras.layers.GlobalMaxPooling3D(name='AvgPool_new')(conv4)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x2 = tf.keras.layers.Dropout(0.1, name='Dropout_new')(x1)

    output0 = tf.keras.layers.Dense(4, use_bias=True,activation='softmax',name='P')(x2)

    model0 = tf.keras.models.Model(inputs =[inputs], outputs = [output0])

    model0.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss="sparse_categorical_crossentropy",metrics=["acc"])
    return model0

#model_no_se = resnet3DClassifier(input_shape=(10, 256, 384, 1), learningRate=0.0001, use_se_module=False)

# Define the model with the SE module
model_with_se = resnet3DClassifier(input_shape=(10, 256, 384, 1), learningRate=0.0001, use_se_module=True)

import cv2
#batch generator for training
def generate_data(train_set, batch_size,shuffle=True):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    train_ID=train_set.imagename.values
    Y_label =train_set.label_int.values
    batch_index=0
    while True:
        image_batch = np.zeros((batch_size,10,IMG_HEIGHT,IMG_WIDTH,1))
        Y_batch0=[]
        Y_batch1=[]

        for b in range(batch_size):
            if i == len(train_ID):
                i = 0
                #shuffle if u want to
                if shuffle:
                  train_set = train_set.sample(frac=1).reset_index(drop=True)
                train_ID=train_set.imagename.values
                Y_label=train_set.label_int.values
            sample = train_ID[i]
            new_image = np.load(sample)

            for band in range(10):
                image_batch[b,band,:,:,0] = cv2.resize(new_image[:,:,band],(IMG_WIDTH,IMG_HEIGHT),cv2.INTER_LINEAR)
            Y_batch0.append(Y_label[i])
            i += 1

        batch_index=batch_index+1
        image_batch=np.array(image_batch)
        Y_batch0= np.array(Y_batch0)

        yield  image_batch, Y_batch0
#model_no_se = resnet3DClassifier(input_shape=(10, 256, 384, 1), learningRate=0.0001, use_se_module=False)
model_with_se = resnet3DClassifier(input_shape=(10, 256, 384, 1), learningRate=0.0001, use_se_module=True)
from sklearn.metrics import classification_report
def test():
  image_batch = np.zeros((1,10,IMG_HEIGHT,IMG_WIDTH,1))

  train_ID = X_test["imagename"].values
  Y_label = X_test["label_int"].values
  Y_batch0 = []

  predicted_label = []
  for i in range(len(X_test)):
    sample = train_ID[i]
    new_image = np.load(sample)

    for band in range(10):
        image_batch[0,band,:,:,0] = cv2.resize(new_image[:,:,band],(IMG_WIDTH,IMG_HEIGHT),cv2.INTER_LINEAR)

    predict = model_with_se.predict(image_batch)

    predicted_label.append(np.argmax(predict[0]))
    Y_batch0.append(Y_label[i])


    print(classification_report(Y_batch0,predicted_label))
# Define both models without and with SE module
#model_no_se = resnet3DClassifier(input_shape=(10, 256, 384, 1), learningRate=0.0001, use_se_module=False)
model_with_se = resnet3DClassifier(input_shape=(10, 256, 384, 1), learningRate=0.0001, use_se_module=True)

# Define callbacks
#filepath_no_se = "Weights-resnet3D_SqueezeClassifier-HyperSpectral-Recognition_no_SE_32.hdf5"
#checkpoint_no_se = tf.keras.callbacks.ModelCheckpoint(filepath_no_se, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
#reduce_lr_no_se = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
#callbacks_list_no_se = [checkpoint_no_se, reduce_lr_no_se]

filepath_with_se = "Weights-resnet3D_SqueezeClassifier-HyperSpectral-Recognition_with_SE_32.hdf5"
checkpoint_with_se = tf.keras.callbacks.ModelCheckpoint(filepath_with_se, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
reduce_lr_with_se = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
callbacks_list_with_se = [checkpoint_with_se, reduce_lr_with_se]

# Batch size
BATCH_SIZE = 3

# Train the model without SE module
#hist_no_se = model_no_se.fit_generator(generate_data(X_train, BATCH_SIZE, True),
#                                      steps_per_epoch=round(len(X_train) / BATCH_SIZE),
#                                      epochs=20,
#                                      validation_data=generate_data(X_test, BATCH_SIZE),
 #                                      validation_steps=round(len(X_test) / BATCH_SIZE),
  #                                     verbose=1,
   #                                    callbacks=callbacks_list_no_se)

# Test the model without SE module
#test()

#Train the model with SE module
hist_with_se = model_with_se.fit_generator(generate_data(X_train, BATCH_SIZE, True),
                                           steps_per_epoch=round(len(X_train) / BATCH_SIZE),
                                           epochs=20,
                                           validation_data=generate_data(X_test, BATCH_SIZE),
                                           validation_steps=round(len(X_test) / BATCH_SIZE),
                                           verbose=1,
                                           callbacks=callbacks_list_with_se)

# Test the model with SE module
test()

