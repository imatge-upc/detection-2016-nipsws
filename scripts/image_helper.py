from keras.preprocessing import image
import numpy as np


def get_all_ids(annotations):
    all_ids = []
    for i in range(len(annotations)):
        all_ids.append(get_ids_objects_from_annotation(annotations[i]))
    return all_ids


def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def get_all_images_pool(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def load_images_names_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    if data_set_name.startswith("aeroplane") | data_set_name.startswith("bird") | data_set_name.startswith("cow"):
        return [x.split(None, 1)[0] for x in image_names]
    else:
        return [x.strip('\n') for x in image_names]


def load_images_labels_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    images_names = f.readlines()
    images_names = [x.split(None, 1)[1] for x in images_names]
    images_names = [x.strip('\n') for x in images_names]
    return images_names


def mask_image_with_mean_background(mask_object_found, image):
    new_image = image
    size_image = np.shape(mask_object_found)
    for j in range(size_image[0]):
        for i in range(size_image[1]):
            if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 103.939
                    new_image[j, i, 1] = 116.779
                    new_image[j, i, 2] = 123.68
    return new_image