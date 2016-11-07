import cv2, numpy as np
import time
import math as mth
from PIL import Image, ImageDraw, ImageFont
import scipy.io
from keras.models import Sequential
from keras import initializations
from keras.initializations import normal, identity
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Adam
import random
from scipy import ndimage
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder

from features import get_image_descriptor_for_image, obtain_compiled_vgg_16, vgg_16, \
    get_conv_image_descriptor_for_image, calculate_all_initial_feature_maps
from parse_xml_annotations import *
from image_helper import *
from metrics import *
from visualization import *
from reinforcement import *

if __name__ == "__main__":

    ######## PATHS definition ########

    # path of pascal voc test
    path_voc_test = "./VOC2007_test/"
    # model name of the weights
    model_name = "model_pool45_crops"
    # path of folder where the weights are
    weights_path = "../models_pool45_crops/"
    # path of where to store visualizations of search sequences
    path_testing_folder = '../testing/'
    # path of VGG16 weights
    path_vgg = "../vgg16_weights.h5"

    ######## MODELS ########

    model_vgg = get_convolutional_vgg16_compiled(path_vgg)
    model = get_q_network(weights_path + model_name)

    ######## LOAD IMAGE NAMES ########

    image_names = np.array([load_images_names_in_data_set('aeroplane_test', path_voc_test)])
    labels = load_images_labels_in_data_set('aeroplane_test', path_voc_test)

    ######## LOAD IMAGES ########
    
    images = get_all_images(image_names, path_voc_test)

    ######## PARAMETERS ########

    # Class category of PASCAL that the RL agent will be searching
    class_object = 1
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3)/4
    scale_mask = float(1)/(scale_subregion*4)
    # Number of steps that the agent does at each image
    number_of_steps = 10
    # Only search first object
    only_first_object = 1

    for j in range(np.size(image_names)):
        if labels[j] == "1":
            image = np.array(images[j])
            # init drawing for visualization
            background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
            draw = ImageDraw.Draw(background)
            image_name = image_names[0][j]
            # get feature maps for the image
            feature_maps = get_feature_maps(model_vgg, image)
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_test)
            gt_masks = generate_bounding_box_from_annotation(annotation, image.shape)
            array_classes_gt_objects = get_ids_objects_from_annotation(annotation)
            size_mask = (image.shape[0], image.shape[1])
            original_shape = size_mask
            image_for_search = image
            region_mask = np.ones([image.shape[0], image.shape[1]])
            count = 0
            # offset of the region observed at each time step
            offset = (0, 0)
            # absolute status is a boolean we indicate if the agent will continue
            # searching object or not. If the first object already covers the whole
            # image, we can put it at 0 so we do not further search there
            absolute_status = 1
            action = 0
            step = 0
            qval = 0
            region_image = image_for_search
            region_mask = np.ones([image.shape[0], image.shape[1]])
            while (step < number_of_steps) and (absolute_status == 1):
                iou = 0
                history_vector = np.zeros([24])
                region_coordinates = np.array([offset[0], offset[1], size_mask[0], size_mask[1]])
                region_descriptor = obtain_descriptor_from_feature_map(feature_maps, region_coordinates)
                region_descriptor_2 = np.reshape(region_descriptor, (25088, 1))
                state = get_state_pool45(history_vector, region_descriptor_2)
                status = 1
                iou = 0
                draw_sequences_test(step, action, qval, draw, region_image, background, path_testing_folder,
                                    region_mask, image_name, bool_draw)
                size_mask = (image.shape[0], image.shape[1])
                original_shape = size_mask
                region_mask = np.ones([image.shape[0], image.shape[1]])
                while (status == 1) & (step < number_of_steps):
                    step += 1
                    qval = model.predict(state.T, batch_size=1)
                    action = (np.argmax(qval))+1
                    # movement action, make the proper zoom on the image
                    if action != 6:
                        region_mask = np.zeros(original_shape)
                        size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
                        if action == 1:
                            offset_aux = (0, 0)
                        elif action == 2:
                            offset_aux = (0, size_mask[1] * scale_mask)
                            offset = (offset[0], offset[1] + size_mask[1] * scale_mask)
                        elif action == 3:
                            offset_aux = (size_mask[0] * scale_mask, 0)
                            offset = (offset[0] + size_mask[0] * scale_mask, offset[1])
                        elif action == 4:
                            offset_aux = (size_mask[0] * scale_mask,
                                          size_mask[1] * scale_mask)
                            offset = (offset[0] + size_mask[0] * scale_mask,
                                      offset[1] + size_mask[1] * scale_mask)
                        elif action == 5:
                            offset_aux = (size_mask[0] * scale_mask / 2,
                                          size_mask[0] * scale_mask / 2)
                            offset = (offset[0] + size_mask[0] * scale_mask / 2,
                                      offset[1] + size_mask[0] * scale_mask / 2)
                        region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                                       offset_aux[1]:offset_aux[1] + size_mask[1]]
                        region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
                    draw_sequences_test(step, action, qval, draw, region_image, background, path_testing_folder,
                                        region_mask, image_name, bool_draw)
                    # trigger action
                    if action == 6:
                        offset = (0, 0)
                        status = 0
                        if step == 1:
                            absolute_status = 0
                        if only_first_object == 1:
                            absolute_status = 0
                        image_for_search = mask_image_with_mean_background(region_mask, image_for_search)
                        region_image = image_for_search
                        feature_maps = get_feature_maps(model_vgg, region_image)
                    history_vector = update_history_vector(history_vector, action)
                    region_coordinates = np.array([offset[0], offset[1], size_mask[0], size_mask[1]])
                    region_descriptor = obtain_descriptor_from_feature_map(feature_maps, region_coordinates)
                    region_descriptor_2 = np.reshape(region_descriptor, (25088, 1))
                    state = get_state_pool45(history_vector, region_descriptor_2)
