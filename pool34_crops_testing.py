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
    # VOC_test_path='/gpfs/projects/bsc31/bsc31429/VOC2007'
    # VOC_test_path = '/gpfs/projects/bsc31/bsc31429/VOC2007_test'
    voc_test_path = './VOCdevkit/VOC2007'
    # VOC_test_path = "/imatge/mbellver/workspace/matlab/o2p/o2p-release1/VOC_experiment/VOC_UCM/"
    # VOC_train_path = '/gpfs/projects/bsc31/bsc31429/VOC2012_train'
    # font_path="/usr/share/fonts/liberation/LiberationMono-Regular.ttf"
    font_path = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
    weights_path = "models_planes_5/"
    model_name = "model0_epoch_33h5"
    path_vgg = '/imatge/mbellver/workspace/matlab/o2p/o2p-release1/VOC_experiment/vgg16_weights.h5'
    # weights_path="./"
    # model_name="model0h5"
    path_testing_folder = "./test_experiments/"
    model_vgg = obtain_compiled_vgg_16(path_vgg)
    labels = load_images_labels_in_data_set('aeroplane_test', voc_test_path)
    final_bbs = []
    image_names = np.array([load_images_names_in_data_set('aeroplane_test', voc_test_path)])
    # imageNames = np.array([loadImagesNamesInDataset('aeroplane_test', VOC_test_path)])
    font = ImageFont.truetype(font_path, 25)
    model = get_q_network(weights_path + model_name)
    images = get_all_images(image_names, voc_test_path)
    class_object = 1
    absolute_status = 1
    total_steps = [[] for _ in xrange(np.size(image_names))]
    count_aux = 0
    bool_draw = 1
    scale_reduction = float(3) / 4
    for j in range(np.size(image_names)):
        if labels[j] == "1":
            count_aux += 1
            image = np.array(images[j])
            final_bbs = []
            background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
            draw = ImageDraw.Draw(background)
            image_name = image_names[0][j]
            print(image_name)
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, voc_test_path)
            gt_masks = generate_bounding_box_from_annotation(annotation, image.shape)
            array_classes_gt_objects = get_ids_objects_from_annotation(annotation)
            size_mask = (image.shape[0], image.shape[1])
            original_shape = size_mask
            image_for_search = image
            region_mask = np.ones([image.shape[0], image.shape[1]])
            count = 0
            offset = (0, 0)
            absolute_status = 1
            in_between_pictures = 20
            action = 0
            step = 0
            qval = 0
            number_of_steps = 10
            region_image = image_for_search
            region_mask = np.ones([image.shape[0], image.shape[1]])
            while (step < number_of_steps) and (absolute_status == 1):
                iou = 0
                history_vector = np.zeros([24])
                state = get_state(region_image, history_vector, model_vgg)
                status = 1
                draw_sequences_test(step, action, qval, draw, font, region_image, background, path_testing_folder,
                                    region_mask, image_name, bool_draw)
                size_mask = (image.shape[0], image.shape[1])
                original_shape = size_mask
                region_mask = np.ones([image.shape[0], image.shape[1]])
                while (status == 1) & (step < number_of_steps):
                    step += 1
                    qval = model.predict(state.T, batch_size=1)
                    action = (np.argmax(qval))+1
                    if action != 6:
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
                    draw_sequences_test(step, action, qval, draw, font, region_image, background, path_testing_folder,
                                        region_mask, image_name, bool_draw)
                    if action == 6:
                        final_bbs.append(region_mask)
                        total_steps[j].append(step)
                        offset = (0, 0)
                        status = 0
                        absolute_status = 0
                        count += 1
                        image_for_search = mask_image_with_mean_background(region_mask, image_for_search)
                        region_image = image_for_search
                    history_vector = update_history_vector(history_vector, action)
                    new_state = get_state(region_image, history_vector, model_vgg)
                    state = new_state
    np.save("number_steps", total_steps)
