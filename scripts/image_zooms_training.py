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
import argparse
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


# Read number of epoch to be trained, to make checkpointing
parser = argparse.ArgumentParser(description='Epoch:')
parser.add_argument("-n", metavar='N', type=int, default=0)
args = parser.parse_args()
epochs_id = int(args.n)


if __name__ == "__main__":

    ######## PATHS definition ########

    # path of PASCAL VOC 2012 or other database to use for training
    path_voc = "./VOC2012/"
    # path of other PASCAL VOC dataset, if you want to train with 2007 and 2012 train datasets
    path_voc2 = "./VOC2007/"
    # path of where to store the models
    path_model = "../models_image_zooms"
    # path of where to store visualizations of search sequences
    path_testing_folder = '../testing_visualizations'
    # path of VGG16 weights
    path_vgg = "../vgg16_weights.h5"

    ######## PARAMETERS ########

    # Class category of PASCAL that the RL agent will be searching
    class_object = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3)/4
    scale_mask = float(1)/(scale_subregion*4)
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 0
    # How many steps can run the agent until finding one object
    number_of_steps = 10
    # Boolean to indicate if you want to use the two databases, or just one
    two_databases = 0
    epochs = 50
    gamma = 0.90
    epsilon = 1
    batch_size = 100
    # Pointer to where to store the last experience in the experience replay buffer,
    # actually there is a pointer for each PASCAL category, in case all categories
    # are trained at the same time
    h = np.zeros([20])
    # Each replay memory (one for each possible category) has a capacity of 100 experiences
    buffer_experience_replay = 1000
    # Init replay memories
    replay = [[] for i in range(20)]
    reward = 0

    ######## MODELS ########

    model_vgg = obtain_compiled_vgg_16(path_vgg)

    # If you want to train it from first epoch, first option is selected. Otherwise,
    # when making checkpointing, weights of last stored weights are loaded for a particular class object

    if epochs_id == 0:
        models = get_array_of_q_networks_for_pascal("0", class_object)
    else:
        models = get_array_of_q_networks_for_pascal(path_model, class_object)

    ######## LOAD IMAGE NAMES ########

    if two_databases == 1:
        image_names1 = np.array([load_images_names_in_data_set('trainval', path_voc)])
        image_names2 = np.array([load_images_names_in_data_set('trainval', path_voc2)])
        image_names = np.concatenate([image_names1, image_names2])
    else:
        image_names = np.array([load_images_names_in_data_set('trainval', path_voc)])

    ######## LOAD IMAGES ########

    if two_databases == 1:
        images1 = get_all_images(image_names1, path_voc)
        images2 = get_all_images(image_names2, path_voc2)
        images = np.concatenate([images1, images2])
    else:
        images = get_all_images(image_names, path_voc)

    for i in range(epochs_id, epochs_id + epochs):
        for j in range(np.size(image_names)):
            masked = 0
            not_finished = 1
            image = np.array(images[j])
            image_name = image_names[0][j]
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
            if two_databases == 1:
                if j < np.size(image_names1):
                    annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
                else:
                    annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc2)
            gt_masks = generate_bounding_box_from_annotation(annotation, image.shape)
            array_classes_gt_objects = get_ids_objects_from_annotation(annotation)
            region_mask = np.ones([image.shape[0], image.shape[1]])
            shape_gt_masks = np.shape(gt_masks)
            available_objects = np.ones(np.size(array_classes_gt_objects))
            # Iterate through all the objects in the ground truth of an image
            for k in range(np.size(array_classes_gt_objects)):
                # Init visualization
                background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
                draw = ImageDraw.Draw(background)
                # We check whether the ground truth object is of the target class category
                if array_classes_gt_objects[k] == class_object:
                    gt_mask = gt_masks[:, :, k]
                    step = 0
                    new_iou = 0
                    # this matrix stores the IoU of each object of the ground-truth, just in case
                    # the agent changes of observed object
                    last_matrix = np.zeros([np.size(array_classes_gt_objects)])
                    region_image = image
                    offset = (0, 0)
                    size_mask = (image.shape[0], image.shape[1])
                    original_shape = size_mask
                    old_region_mask = region_mask
                    region_mask = np.ones([image.shape[0], image.shape[1]])
                    # If the ground truth object is already masked by other already found masks, do not
                    # use it for training
                    if masked == 1:
                        for p in range(gt_masks.shape[2]):
                            overlap = calculate_overlapping(old_region_mask, gt_masks[:, :, p])
                            if overlap > 0.60:
                                available_objects[p] = 0
                    # We check if there are still obejcts to be found
                    if np.count_nonzero(available_objects) == 0:
                        not_finished = 0
                    # follow_iou function calculates at each time step which is the groun truth object
                    # that overlaps more with the visual region, so that we can calculate the rewards appropiately
                    iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask, array_classes_gt_objects,
                                                                  class_object, last_matrix, available_objects)
                    new_iou = iou
                    gt_mask = gt_masks[:, :, index]
                    # init of the history vector that indicates past actions (6 actions * 4 steps in the memory)
                    history_vector = np.zeros([24])
                    # computation of the initial state
                    state = get_state(region_image, history_vector, model_vgg)
                    # status indicates whether the agent is still alive and has not triggered the terminal action
                    status = 1
                    action = 0
                    reward = 0
                    if step > number_of_steps:
                        background = draw_sequences(i, k, step, action, draw, region_image, background,
                                                    path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                                                    bool_draw)
                        step += 1
                    while (status == 1) & (step < number_of_steps) & not_finished:
                        category = int(array_classes_gt_objects[k]-1)
                        model = models[0][category]
                        qval = model.predict(state.T, batch_size=1)
                        background = draw_sequences(i, k, step, action, draw, region_image, background,
                                                    path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                                                    bool_draw)
                        step += 1
                        # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
                        if (i < 100) & (new_iou > 0.5):
                            action = 6
                        # epsilon-greedy policy
                        elif random.random() < epsilon:
                            action = np.random.randint(1, 7)
                        else:
                            action = (np.argmax(qval))+1
                        # terminal action
                        if action == 6:
                            iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask,
                                                                          array_classes_gt_objects, class_object,
                                                                          last_matrix, available_objects)
                            gt_mask = gt_masks[:, :, index]
                            reward = get_reward_trigger(new_iou)
                            background = draw_sequences(i, k, step, action, draw, region_image, background,
                                                        path_testing_folder, iou, reward, gt_mask, region_mask,
                                                        image_name, bool_draw)
                            step += 1
                        # movement action, we perform the crop of the corresponding subregion
                        else:
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
                            iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask,
                                                                          array_classes_gt_objects, class_object,
                                                                          last_matrix, available_objects)
                            gt_mask = gt_masks[:, :, index]
                            reward = get_reward_movement(iou, new_iou)
                            iou = new_iou
                        history_vector = update_history_vector(history_vector, action)
                        new_state = get_state(region_image, history_vector, model_vgg)
                        # Experience replay storage
                        if len(replay[category]) < buffer_experience_replay:
                            replay[category].append((state, action, reward, new_state))
                        else:
                            if h[category] < (buffer_experience_replay-1):
                                h[category] += 1
                            else:
                                h[category] = 0
                            h_aux = h[category]
                            h_aux = int(h_aux)
                            replay[category][h_aux] = (state, action, reward, new_state)
                            minibatch = random.sample(replay[category], batch_size)
                            X_train = []
                            y_train = []
                            # we pick from the replay memory a sampled minibatch and generate the training samples
                            for memory in minibatch:
                                old_state, action, reward, new_state = memory
                                old_qval = model.predict(old_state.T, batch_size=1)
                                newQ = model.predict(new_state.T, batch_size=1)
                                maxQ = np.max(newQ)
                                y = np.zeros([1, 6])
                                y = old_qval
                                y = y.T
                                if action != 6: #non-terminal state
                                    update = (reward + (gamma * maxQ))
                                else: #terminal state
                                    update = reward
                                y[action-1] = update #target output
                                X_train.append(old_state)
                                y_train.append(y)
                            X_train = np.array(X_train)
                            y_train = np.array(y_train)
                            X_train = X_train.astype("float32")
                            y_train = y_train.astype("float32")
                            X_train = X_train[:, :, 0]
                            y_train = y_train[:, :, 0]
                            hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)
                            models[0][category] = model
                            state = new_state
                        if action == 6:
                            status = 0
                            masked = 1
                            # we mask object found with ground-truth so that agent learns faster
                            image = mask_image_with_mean_background(gt_mask, image)
                        else:
                            masked = 0
                    available_objects[index] = 0
        if epsilon > 0.1:
            epsilon -= 0.1
        for t in range (np.size(models)):
            if t == (class_object-1):
                string = path_model + '/model' + str(t) + '_epoch_' + str(i) + 'h5'
                string2 = path_model + '/model' + str(t) + 'h5'
                model = models[0][t]
                model.save_weights(string, overwrite=True)
                model.save_weights(string2, overwrite=True)

