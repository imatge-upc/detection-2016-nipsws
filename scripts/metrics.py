import numpy as np
import cv2


def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou


def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_masks, mask, array_classes_gt_objects, object_id, last_matrix, available_objects):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(array_classes_gt_objects)):
        if array_classes_gt_objects[k] == object_id:
            if available_objects[k] == 1:
                gt_mask = gt_masks[:, :, k]
                iou = calculate_iou(mask, gt_mask)
                results[k] = iou
            else:
                results[k] = -1
    max_result = max(results)
    ind = np.argmax(results)
    iou = last_matrix[ind]
    new_iou = max_result
    return iou, new_iou, results, ind
