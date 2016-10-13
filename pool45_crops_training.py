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



# ACTIONS
# UP-LEFT 1
# UP-RIGHT 2
# DOWN-LEFT 3
# DOWN-RIGHT 4
# CENTER 5
# TRIGGER 6

parser = argparse.ArgumentParser(description='Epoch:')
parser.add_argument("-n", metavar='N', type=int)
args = parser.parse_args()
epochs_id=int(args.n)


def get_state_2(history_vector,  region_descriptor):
    history_vector = np.reshape(history_vector, (24, 1))
    return np.vstack((region_descriptor, history_vector))


if __name__ == "__main__":

    path_font = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
    path_voc = "/imatge/mbellver/workspace/matlab/o2p/o2p-release1/VOC_experiment/VOC_UCM/"
    path_voc2 = "./VOCdevkit/VOC2007/"
    path_vgg = '/imatge/mbellver/workspace/matlab/o2p/o2p-release1/VOC_experiment/vgg16_weights.h5'
    # path_model="models_planes_4/"
    # path_testing_folder = './fast_testing'

    # PATHS FOR MINOTAURE-BSC
    # path_font="/usr/share/fonts/liberation/LiberationMono-Regular.ttf"
    # path_VOC="/gpfs/projects/bsc31/bsc31429/VOC2012_train/"
    # TO DO, if we wan to add another database
    # path_VOC2="/gpfs/projects/bsc31/bsc31429/VOC2007_train/"
    # path_VOC_test="/gpfs/projects/bsc31/bsc31429/VOC2012_test/"
    # path_model = "models_planes_5/"
    path_model = "models_example"
    path_testing_folder = './fast_testing'
    model_vgg = get_convolutional_vgg16_compiled(path_vgg)
    font = ImageFont.truetype(path_font, 30)
    if epochs_id == 0:
        models = get_array_of_q_networks_for_pascal("0")
    else:
        models = get_array_of_q_networks_for_pascal(path_model)
    # image_names=np.array([load_images_names_in_data_set('aeroplane_trainval', path_VOC)])
    # TODO: Uncomment if you want to use more than one database
    image_names1 = np.array([load_images_names_in_data_set('aeroplane_trainval', path_voc)])
    image_names2 = np.array([load_images_names_in_data_set('aeroplane_trainval', path_voc2)])
    image_names = np.concatenate([image_names1, image_names2], axis=1)

    # images=get_all_images(image_mames, path_voc)
    # TODO: Uncomment if you want to use more than one database
    images1 = get_all_images(image_names1, path_voc)
    images2 = get_all_images(image_names2, path_voc2)
    images = images1 + images2

    initial_feature_maps1 = calculate_all_initial_feature_maps(images, model_vgg, image_names1)
    initial_feature_maps2 = calculate_all_initial_feature_maps(images, model_vgg, image_names2)
    initial_feature_maps = initial_feature_maps1 + initial_feature_maps2

    reward = 0
    bool_draw = 0
    scale_reduction = float(3)/4
    epochs = 3000
    epochs_batch = 100
    gamma = 0.90
    epsilon = 1
    batch_size = 100
    h = np.zeros([20])
    counter = np.zeros([20])
    buffer = 100
    number_of_steps = 10
    replay = [[] for i in range(1000)]
    allLosses = np.zeros([epochs, 20])
    class_object = 1
    for i in range(epochs_id, epochs_id+epochs_batch):
        print(i)
        for j in range(np.size(image_names)):
            masked = 0
            not_finished = 1
            image = np.array(images[j])
            image_name = image_names[0][j]
            feature_maps = initial_feature_maps[j]
            # annotation = getBBofGTFromPascalXMLAnnotation(imageName, path_VOC)
            # TODO:Uncomment for working with two databases
            if j < np.size(image_names1):
                annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
            else:
                annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc2)
            gt_masks = generate_bounding_box_from_annotation(annotation, image.shape)
            array_classes_gt_objects = get_ids_objects_from_annotation(annotation)
            region_mask = np.ones([image.shape[0], image.shape[1]])
            # gtMasks=np.array(allgtmasks[j])
            shape_gt_masks = np.shape(gt_masks)
            available_objects = np.ones(np.size(array_classes_gt_objects))
            # if (gtMasks.ndim==3):
            #     arrayClassesGTObjects=arrayClassesGTObjects[:][0:(shapeGTMasks[2]-1)]
            for k in range(np.size(array_classes_gt_objects)):
                background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
                draw = ImageDraw.Draw(background)
                if array_classes_gt_objects[k] == class_object:
                    gt_mask = gt_masks[:, :, k]
                    step = 0
                    reward = 0
                    last_matrix = np.zeros([np.size(array_classes_gt_objects)])
                    new_iou = 0
                    region_image = image
                    offset = (0, 0)
                    size_mask = (image.shape[0], image.shape[1])
                    original_shape = size_mask
                    old_region_mask = region_mask
                    region_mask=np.ones([image.shape[0], image.shape[1]])
                    # IoU=calculateIoU(region_mask,gt_mask)
                    # print(gtMasks.shape)
                    if masked == 1:
                        for p in range(gt_masks.shape[2]):
                            overlap = calculate_overlapping(old_region_mask, gt_masks[:, :, p])
                            if overlap > 0.35:
                                available_objects[p] = 0
                    if np.count_nonzero(available_objects) == 0:
                        not_finished = 0
                    iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask, array_classes_gt_objects,
                                                                  class_object, last_matrix, available_objects)
                    new_iou = iou
                    gt_mask = gt_masks[:, :, index]
                    history_vector = np.zeros([24])
                    region_coordinates = np.array([offset[0], offset[1], size_mask[0], size_mask[1]])
                    region_descriptor = obtain_descriptor_from_feature_map(feature_maps, region_coordinates)
                    region_descriptor_2 = np.reshape(region_descriptor, (25088, 1))
                    state = get_state_2(history_vector, region_descriptor_2)
                    # state=getState(regionImage,history_vector,model_vgg)
                    status = 1
                    action = 0
                    if step > number_of_steps:
                        background = draw_sequences(i, k, step, action, draw, font, region_image, background,
                                                    path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                                                    bool_draw)
                        step += 1
                    while (status == 1) & (step < number_of_steps) & not_finished:
                        category = int(array_classes_gt_objects[k]-1)
                        counter[category] += 1
                        model = models[0][category]
                        initial1 = time.time()
                        qval = model.predict(state.T, batch_size=1)
                        final1 = time.time()
                        print("Prediction")
                        print(final1 - initial1)
                        background = draw_sequences(i, k, step, action, draw, font, region_image, background,
                                                    path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                                                    bool_draw)
                        step += 1
                        if (i < 100) & (new_iou > 0.5):
                            action = 6
                        elif random.random() < epsilon:
                            action = np.random.randint(1, 7)
                        else:
                            action = (np.argmax(qval))+1
                        if action == 6:
                            iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask,
                                                                          array_classes_gt_objects, class_object,
                                                                          last_matrix, available_objects)
                            gt_mask = gt_masks[:, :, index]
                            reward = get_reward_trigger(new_iou)
                            background = draw_sequences(i, k, step, action, draw, font, region_image, background,
                                                        path_testing_folder, iou, reward, gt_mask, region_mask,
                                                        image_name, bool_draw)
                            step += 1
                        else:
                            initial1 = time.time()
                            region_mask = np.zeros(original_shape)
                            size_mask = (size_mask[0] * scale_reduction, size_mask[1] * scale_reduction)
                            if action == 1:
                                offset_aux = (0, 0)
                            elif action == 2:
                                offset_aux = (0, size_mask[1] * (1 - scale_reduction))
                                offset = (offset[0], offset[1] + size_mask[1] * (1 - scale_reduction))
                            elif action == 3:
                                offset_aux = (size_mask[0] * (1 - scale_reduction), 0)
                                offset = (offset[0] + size_mask[0] * (1 - scale_reduction), offset[1])
                            elif action == 4:
                                offset_aux = (size_mask[0] * (1 - scale_reduction),
                                              size_mask[1] * (1 - scale_reduction))
                                offset = (offset[0] + size_mask[0] * (1 - scale_reduction),
                                          offset[1] + size_mask[1] * (1 - scale_reduction))
                            elif action == 5:
                                offset_aux = (size_mask[0] * (1 - scale_reduction) / 2,
                                              size_mask[0] * (1 - scale_reduction) / 2)
                                offset = (offset[0] + size_mask[0] * (1 - scale_reduction) / 2,
                                          offset[1] + size_mask[0] * (1 - scale_reduction) / 2)
                            region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                                           offset_aux[1]:offset_aux[1] + size_mask[1]]
                            region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
                            # new_IoU=calculateIoU(region_mask,gt_mask)
                            iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask,
                                                                          array_classes_gt_objects, class_object,
                                                                          last_matrix, available_objects)
                            gt_mask = gt_masks[:, :, index]
                            reward = get_reward_movement(iou, new_iou)
                            iou = new_iou
                            history_vector = update_history_vector(history_vector, action)
                        region_coordinates = np.array([offset[0], offset[1], size_mask[0], size_mask[1]])
                        region_descriptor = obtain_descriptor_from_feature_map(feature_maps, region_coordinates)
                        region_descriptor_2 = np.reshape(region_descriptor,(25088,1))
                        new_state = get_state_2(history_vector, region_descriptor_2)
                        #Experience replay storage
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
                            initial1 = time.time()
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
                                    # print update
                                else: #terminal state
                                    update = reward
                                    # print update
                                y[action-1] = update #target output
                                # print(y)
                                X_train.append(old_state)
                                y_train.append(y)
                            final1 = time.time()
                            print("Replay memory")
                            print (final1-initial1)
                            X_train = np.array(X_train)
                            y_train = np.array(y_train)
                            X_train = X_train.astype("float32")
                            y_train = y_train.astype("float32")
                            X_train = X_train[:, :, 0]
                            y_train = y_train[:, :, 0]
                            initial1 = time.time()
                            hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)
                            aux2 = int(counter[category]-1)
                            aux = hist.history
                            allLosses[i][category]=allLosses[i][category]+aux['loss']
                            models[0][category] = model
                            state = new_state
                        if action == 6:
                            status = 0
                            masked = 1
                            if i < 300:
                                image = mask_image_with_mean_background(gt_mask, image)
                            else:
                                image = mask_image_with_mean_background(region_mask, image)
                        else:
                            masked = 0
                    available_objects[index] = 0
        if epsilon > 0.1:
            epsilon -= 0.01
        for j in range(20):
            if counter[j] != 0:
                allLosses[i][j] = allLosses[i][j]/counter[j]
            else:
                allLosses[i][j] = 0
        scipy.io.savemat('losses_planes24sept.mat', {'losses': allLosses})
        for t in range (np.size(models)):
            if t == (class_object-1):
                string = path_model + '/model' + str(t) + '_epoch_' + str(i) + 'h5'
                string2 = path_model + '/model' + str(t) + 'h5'
                model= models[0][t]
                # if i % 10 == 0:
                model.save_weights(string, overwrite=True)
                model.save_weights(string2, overwrite=True)
        # return (epochs_id + epochs_batch)
    # second2=time.time()
    # print ('\n TOTAL TIME \n')
    # print (second2-initial1)
