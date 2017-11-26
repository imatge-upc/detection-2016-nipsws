import xml.etree.ElementTree as ET
import numpy as np


def get_bb_of_gt_from_pascal_xml_annotation(xml_name, voc_path):
    string = voc_path + '/Annotations/' + xml_name + '.xml'
    tree = ET.parse(string)
    root = tree.getroot()
    names = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    for child in root:
        if child.tag == 'object':
            for child2 in child:
                if child2.tag == 'name':
                    names.append(child2.text)
                elif child2.tag == 'bndbox':
                    for child3 in child2:
                        if child3.tag == 'xmin':
                            x_min.append(child3.text)
                        elif child3.tag == 'xmax':
                            x_max.append(child3.text)
                        elif child3.tag == 'ymin':
                            y_min.append(child3.text)
                        elif child3.tag == 'ymax':
                            y_max.append(child3.text)
    category_and_bb = np.zeros([np.size(names), 5])
    for i in range(np.size(names)):
        category_and_bb[i][0] = get_id_of_class_name(names[i])
        category_and_bb[i][1] = x_min[i]
        category_and_bb[i][2] = x_max[i]
        category_and_bb[i][3] = y_min[i]
        category_and_bb[i][4] = y_max[i]
    return category_and_bb


def get_all_annotations(image_names, voc_path):
    annotations = []
    for i in range(np.size(image_names)):
        image_name = image_names[0][i]
        annotations.append(get_bb_of_gt_from_pascal_xml_annotation(image_name, voc_path))
    return annotations


def generate_bounding_box_from_annotation(annotation, image_shape):
    length_annotation = annotation.shape[0]
    masks = np.zeros([image_shape[0], image_shape[1], length_annotation])
    for i in range(0, length_annotation):
        masks[annotation[i, 3]:annotation[i, 4], annotation[i, 1]:annotation[i, 2], i] = 1
    return masks


def get_ids_objects_from_annotation(annotation):
    return annotation[:, 0]


def get_id_of_class_name (class_name):
    if class_name == 'aeroplane':
        return 1
    elif class_name == 'bicycle':
        return 2
    elif class_name == 'bird':
        return 3
    elif class_name == 'boat':
        return 4
    elif class_name == 'bottle':
        return 5
    elif class_name == 'bus':
        return 6
    elif class_name == 'car':
        return 7
    elif class_name == 'cat':
        return 8
    elif class_name == 'chair':
        return 9
    elif class_name == 'cow':
        return 10
    elif class_name == 'diningtable':
        return 11
    elif class_name == 'dog':
        return 12
    elif class_name == 'horse':
        return 13
    elif class_name == 'motorbike':
        return 14
    elif class_name == 'person':
        return 15
    elif class_name == 'pottedplant':
        return 16
    elif class_name == 'sheep':
        return 17
    elif class_name == 'sofa':
        return 18
    elif class_name == 'train':
        return 19
    elif class_name == 'tvmonitor':
        return 20
















