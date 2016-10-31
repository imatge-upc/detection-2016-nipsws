from PIL import Image, ImageDraw, ImageFont
import numpy as np

path_font = "/usr/share/fonts/liberation/LiberationMono-Regular.ttf"
font = ImageFont.truetype(path_font, 24)


def string_for_action(action):
    if action == 0:
        return "START"
    if action == 1:
        return 'up-left'
    elif action == 2:
        return 'up-right'
    elif action == 3:
        return 'down-left'
    elif action == 4:
        return 'down-right'
    elif action == 5:
        return 'center'
    elif action == 6:
        return 'TRIGGER'


def draw_sequences(i, k, step, action, draw, region_image, background, path_testing_folder, iou, reward,
                   gt_mask, region_mask, image_name, save_boolean):
    mask = Image.fromarray(255 * gt_mask)
    mask_img = Image.fromarray(255 * region_mask)
    image_offset = (1000 * step, 70)
    text_offset = (1000 * step, 550)
    masked_image_offset = (1000 * step, 1400)
    mask_offset = (1000 * step, 700)
    action_string = string_for_action(action)
    footnote = 'action: ' + action_string + ' ' + 'reward: ' + str(reward) + ' Iou:' + str(iou)
    draw.text(text_offset, str(footnote), (0, 0, 0), font=font)
    img_for_paste = Image.fromarray(region_image)
    background.paste(img_for_paste, image_offset)
    background.paste(mask, mask_offset)
    background.paste(mask_img, masked_image_offset)
    file_name = path_testing_folder + '/' + image_name + str(i) + '_object_' + str(k) + '.png'
    if save_boolean == 1:
        background.save(file_name)
    return background


def draw_sequences_test(step, action, qval, draw, region_image, background, path_testing_folder,
                        region_mask, image_name, save_boolean):
    aux = np.asarray(region_image, np.uint8)
    img_offset = (1000 * step, 70)
    footnote_offset = (1000 * step, 550)
    q_predictions_offset = (1000 * step, 500)
    mask_img_offset = (1000 * step, 700)
    img_for_paste = Image.fromarray(aux)
    background.paste(img_for_paste, img_offset)
    mask_img = Image.fromarray(255 * region_mask)
    background.paste(mask_img, mask_img_offset)
    footnote = 'action: ' + str(action)
    q_val_predictions_text = str(qval)
    draw.text(footnote_offset, footnote, (0, 0, 0), font=font)
    draw.text(q_predictions_offset, q_val_predictions_text, (0, 0, 0), font=font)
    file_name = path_testing_folder + image_name + '.png'
    if save_boolean == 1:
        background.save(file_name)
    return background


