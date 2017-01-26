from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
import cv2, numpy as np
import math
import numpy, scipy
from scipy import interpolate
import scipy.ndimage
import time

# the feature size is of 7x7xp, being p the number of channels
feature_size = 7
# the relative scale reduction of the shallower feature map compared to the initial image input
scale_reduction_shallower_feature = 16
# the relative scale reduction of the deeper feature map compared to the initial image input
scale_reduction_deeper_feature = 32
# scaling of the input image
factor_x_input = float(1)
factor_y_input = float(1)


# Interpolation of 2d features for a single channel of a feature map
def interpolate_2d_features(features):
    out_size = feature_size
    x = np.arange(features.shape[0])
    y = np.arange(features.shape[1])
    z = features
    xx = np.linspace(x.min(), x.max(), out_size)
    yy = np.linspace(y.min(), y.max(), out_size)
    new_kernel = interpolate.RectBivariateSpline(x, y, z, kx=1, ky=1)
    kernel_out = new_kernel(xx, yy)
    return kernel_out


# Interpolation 2d of each channel, so we obtain 3d interpolated feature maps
def interpolate_3d_features(features):
    new_features = np.zeros([512, feature_size, feature_size])
    for i in range(features.shape[0]):
        new_features[i, :, :] = interpolate_2d_features(features[i, :, :])
    return new_features


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
    return model


def get_convolutional_vgg16_compiled(vgg_weights_path):
    model_vgg = obtain_compiled_vgg_16(vgg_weights_path)
    for i in range(0, 6):
        model_vgg = pop_layer(model_vgg)
    return model_vgg


def get_feature_maps(model, img):
    return [get_feature_map_4(model, img), get_feature_map_8(model, img)]


# get deeper feature map
def get_feature_map_8(model, im):
    im = im.astype(np.float32)
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]
        # Zero-center by mean pixel
        im[0, :, :] -= 103.939
        im[1, :, :] -= 116.779
        im[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        # Zero-center by mean pixel
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, model.outputs)
    feature_map = _convout1_f([0] + [im])
    feature_map = np.array([feature_map])
    feature_map = feature_map[0, 0, 0, :, :, :]
    return feature_map


# get shallower feature map
def get_feature_map_4(model, im):
    im = im.astype(np.float32)
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]
        # Zero-center by mean pixel
        im[0, :, :] -= 103.939
        im[1, :, :] -= 116.779
        im[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        # Zero-center by mean pixel
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, [model.layers[23].output])
    feature_map = _convout1_f([0] + [im])
    feature_map = np.array([feature_map])
    feature_map = feature_map[0, 0, 0, :, :, :]
    return feature_map


def crop_roi(feature_map, coordinates):
    return feature_map[:, coordinates[0]:coordinates[0]+coordinates[2], coordinates[1]:coordinates[1]+coordinates[3]]


# this method decides whether to use the deeper or the shallower feature map
# and then crops and interpolates if necessary the features to obtain a final descriptor of 7x7xp
def obtain_descriptor_from_feature_map(feature_maps, region_coordinates):
    initial_width = region_coordinates[2]*factor_x_input
    initial_height = region_coordinates[3]*factor_y_input
    scale_aux = math.sqrt(initial_height*initial_width)/math.sqrt(feature_size*feature_size)
    if scale_aux > scale_reduction_deeper_feature:
        scale = scale_reduction_deeper_feature
        feature_map = feature_maps[1]
    else:
        scale = scale_reduction_shallower_feature
        feature_map = feature_maps[0]
    new_width = initial_width/scale
    new_height = initial_height/scale
    if new_width < feature_size:
        new_width = feature_size
    if new_height < feature_size:
        new_height = feature_size
    xo = region_coordinates[0]/scale
    yo = region_coordinates[1]/scale
    feat = np.array([feature_map])
    if new_width + xo > feat.shape[2]:
        xo = feat.shape[2] - new_width
    if new_height + yo > feat.shape[3]:
        yo = feat.shape[3] - new_height
    if xo < 0:
        xo = 0
    if yo < 0:
        yo = 0
    new_coordinates = np.array([xo, yo, new_width, new_height])
    roi = crop_roi(feature_map, new_coordinates)
    if roi.shape[1] < feature_size & roi.shape[2] < feature_size:
        features = interpolate_3d_features(roi)
    elif roi.shape[2] < feature_size:
        features = interpolate_3d_features(roi)
    elif roi.shape[1] < feature_size:
        features = interpolate_3d_features(roi)
    else:
        features = extract_features_from_roi(roi)
    return features


# ROI-pooling features
def extract_features_from_roi(roi):
    roi_width = roi.shape[1]
    roi_height = roi.shape[2]
    new_width = roi_width / feature_size
    new_height = roi_height / feature_size
    pooled_values = np.zeros([feature_size, feature_size, 512])
    for j in range(512):
        for i in range(feature_size):
            for k in range(feature_size):
                if k == (feature_size-1) & i == (feature_size-1):
                    patch = roi[j, i * new_width:roi_width, k * new_height:roi_height]
                elif k == (feature_size-1):
                    patch = roi[j, i * new_width:(i + 1) * new_width, k * new_height:roi_height]
                elif i == (feature_size-1):
                    patch = roi[j, i * new_width:roi_width, k * new_height:(k + 1) * new_height]
                else:
                    patch = roi[j, i * new_width:(i + 1) * new_width, k * new_height:(k + 1) * new_height]
                pooled_values[i, k, j] = np.max(patch)
    return pooled_values


def calculate_all_initial_feature_maps(images, model, image_names):
    initial_feature_maps = []
    for z in range(np.size(image_names)):
        initial_feature_maps.append(get_feature_maps(model, np.array(images[z])))
    return initial_feature_maps


def get_image_descriptor_for_image(image, model):
    im = cv2.resize(image, (224, 224)).astype(np.float32)
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]
        # Zero-center by mean pixel
        im[0, :, :] -= 103.939
        im[1, :, :] -= 116.779
        im[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        # Zero-center by mean pixel
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, [model.layers[33].output])
    return _convout1_f([0] + [im])


def get_conv_image_descriptor_for_image(image, model):
    im = cv2.resize(image, (224, 224)).astype(np.float32)
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]
        # Zero-center by mean pixel
        im[0, :, :] -= 103.939
        im[1, :, :] -= 116.779
        im[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        # Zero-center by mean pixel
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, [model.layers[31].output])
    return _convout1_f([0] + [im])


def obtain_compiled_vgg_16(vgg_weights_path):
    model = vgg_16(vgg_weights_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def vgg_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

