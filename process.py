import numpy as np
from settings import loader_settings
import medpy.io
import os, pathlib
from medpy.metric import dc , precision, recall
import tensorflow as tf
import math
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import keras
from keras.callbacks import CSVLogger
from keras.models import load_model
import nibabel as nib

import cv2
import numpy as np


from keras.utils import custom_object_scope
import numpy as np
import matplotlib.pyplot as plt

# Load the model
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-10):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = 1 - (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return answer


def convolution_block(x, filters, kernel_size = 3, batchnorm = True):
    x = Conv3D(filters, kernel_size = 3, kernel_initializer = 'he_normal', padding = 'same', data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def residual_block(blockInput, n_filters, batchnorm = False):
    x = convolution_block(blockInput, n_filters, 3, batchnorm)
    x = convolution_block(x, n_filters, 3, batchnorm)
    x = Add()([x, blockInput])
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
class channel_attention(tf.keras.layers.Layer):
    """ 
    channel attention module 
    
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, ratio=8, **kwargs):
        super(channel_attention, self).__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super(channel_attention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(
            channel // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.shared_layer_two = tf.keras.layers.Dense(
            channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'
        )
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling3D()(inputs)
        avg_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling3D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])


class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super(spatial_attention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def get_config(self):
        config = super(spatial_attention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv3D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True)
        )(inputs)
        max_pool = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True)
        )(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)

        return tf.keras.layers.multiply([inputs, feature])


def cbam_block(feature, ratio=8, kernel_size=7):

    feature = channel_attention(ratio=ratio)(feature)
    feature = spatial_attention(kernel_size=kernel_size)(feature)

    return feature
    

def build_model(input_layer, n_filters, DropoutRatio = 0.3, batchnorm = True):
    # downsample layer 1
    conv1 = convolution_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    conv1 = residual_block(conv1, n_filters * 1)
    conv1 = residual_block(conv1, n_filters * 1, True)
    pool1 = MaxPooling3D(pool_size = 2 , data_format = 'channels_last')(conv1)
    pool1 = Dropout(DropoutRatio)(pool1)

    # downsample layer 2
    conv2 = convolution_block(pool1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    conv2 = residual_block(conv2, n_filters * 2)
    conv2 = residual_block(conv2, n_filters * 2, True)
    pool2 = MaxPooling3D(pool_size = 2 , data_format = 'channels_last')(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # downsample layer 3
    conv3 = convolution_block(pool2, n_filters * 3, kernel_size = 3, batchnorm = batchnorm)
    conv3 = residual_block(conv3, n_filters * 3)
    conv3 = residual_block(conv3, n_filters * 3, True)
    pool3 = MaxPooling3D(pool_size = 2 , data_format = 'channels_last')(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    convm = convolution_block(pool3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    convm = residual_block(convm, n_filters * 4)
    convm = residual_block(convm, n_filters * 4, True)
    
    # upsample layer 3
    deconv3 = Conv3DTranspose(n_filters * 3, 3 , strides = 2, padding = 'same', data_format = 'channels_last',kernel_regularizer = regularizers.l2(0.01))(convm)
    deconv3 = concatenate([deconv3, cbam_block(conv3)])
    deconv3 = Dropout(DropoutRatio)(deconv3)
    uconv3 = convolution_block(deconv3, n_filters * 3, kernel_size = 3, batchnorm = batchnorm)

    # upsample layer 2
    deconv2 = Conv3DTranspose(n_filters * 2, 3 , strides = 2, padding = 'same', data_format = 'channels_last',kernel_regularizer = regularizers.l2(0.01))(uconv3)
    deconv2 = concatenate([deconv2, conv2])
    deconv2 = Dropout(DropoutRatio)(deconv2)
    uconv2 = convolution_block(deconv2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    # upsample layer 3
    deconv1 = Conv3DTranspose(n_filters * 1, 3, strides = 2, padding = 'same', data_format = 'channels_last',kernel_regularizer = regularizers.l2(0.01))(uconv2)
    deconv1 = concatenate([deconv1, conv1])
    deconv1 = Dropout(DropoutRatio)(deconv1)
    uconv1 = convolution_block(deconv1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv3D(1, 1, activation='sigmoid')(uconv1)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

class Seg():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return
        
    def process(self):
        inp_path = loader_settings['InputPath']  # Path for the input
        out_path = loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        with custom_object_scope({'spatial_attention': spatial_attention,'channel_attention':channel_attention,'tversky_loss':tversky_loss,'dice_coef':dice_coef}):
            model = load_model("model.h5")
        for fil in file_list:
            dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            im_shape = dat.shape
            dat = dat.reshape(*im_shape)
            tmp2 = np.zeros([192,192,189], dtype = float)
            for j in range(0,189):
                img = dat[:,:,j]
                img_sm=cv2.resize(img, (192,192), interpolation=cv2.INTER_CUBIC)
                tmp2[:, :,j] = img_sm
            test_sample=np.expand_dims(tmp2, -1) 
            ans_slices=[]
            print("Sampling done")
            for i in range(22,166,8):
                test_slice=test_sample[:,:,i:i+8]
                test_slice=test_slice.reshape(1, 192, 192, 8, 1)
                pred = model.predict(test_slice)
                threshold=0.5
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                pred=pred.reshape((192, 192, 8))
                ans_slices.append(pred)
            dat=ans_slices[0]
            for i in range(1,len(ans_slices)):
                dat=np.concatenate((dat,ans_slices[i]),axis=2)
            final_ans=np.zeros((192,192,189))
            final_ans[:,:,22:166]=dat
            ###
            ###########
            ans = np.zeros([197,233,189], dtype = int)
            for j in range(0,189):
                img = final_ans[:,:,j]
                img_sm=cv2.resize(img, (233,197), interpolation=cv2.INTER_NEAREST)
                ans[:, :,j] = img_sm
            dat=ans
            dat = dat.reshape(*im_shape)
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            print(f'=== saving {out_filepath} from {fil} ===')
            medpy.io.save(dat, out_filepath, hdr=hdr)
            print("Done")
        return


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Seg().process()