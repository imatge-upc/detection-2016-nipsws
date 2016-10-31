import numpy as np
from keras.models import Sequential
from keras import initializations
from keras.initializations import normal, identity
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD, Adam
from features import *

# Different actions that the agent can do
number_of_actions = 6
# Actions captures in the history vector
actions_of_history = 4
# Visual descriptor size
visual_descriptor_size = 25088
# Reward movement action
reward_movement_action = 1
# Reward terminal action
reward_terminal_action = 3
# IoU required to consider a positive detection
iou_threshold = 0.5


def update_history_vector(history_vector, action):
    action_vector = np.zeros(number_of_actions)
    action_vector[action-1] = 1
    size_history_vector = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(number_of_actions*actions_of_history)
    if size_history_vector < actions_of_history:
        aux2 = 0
        for l in range(number_of_actions*size_history_vector, number_of_actions*size_history_vector+number_of_actions - 1):
            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector
    else:
        for j in range(0, number_of_actions*(actions_of_history-1) - 1):
            updated_history_vector[j] = history_vector[j+number_of_actions]
        aux = 0
        for k in range(number_of_actions*(actions_of_history-1), number_of_actions*actions_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector


def get_state(image, history_vector, model_vgg):
    descriptor_image = get_conv_image_descriptor_for_image(image, model_vgg)
    descriptor_image = np.reshape(descriptor_image, (visual_descriptor_size, 1))
    history_vector = np.reshape(history_vector, (number_of_actions*actions_of_history, 1))
    state = np.vstack((descriptor_image, history_vector))
    return state


def get_state_pool45(history_vector,  region_descriptor):
    history_vector = np.reshape(history_vector, (24, 1))
    return np.vstack((region_descriptor, history_vector))


def get_reward_movement(iou, new_iou):
    if new_iou > iou:
        reward = reward_movement_action
    else:
        reward = - reward_movement_action
    return reward


def get_reward_trigger(new_iou):
    if new_iou > iou_threshold:
        reward = reward_terminal_action
    else:
        reward = - reward_terminal_action
    return reward


def get_q_network(weights_path):
    model = Sequential()
    model.add(Dense(1024, init=lambda shape, name: normal(shape, scale=0.01, name=name), input_shape=(25112,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('linear'))
    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    if weights_path != "0":
        model.load_weights(weights_path)
    return model


def get_array_of_q_networks_for_pascal(weights_path, class_object):
    q_networks = []
    if weights_path == "0":
        for i in range(20):
            q_networks.append(get_q_network("0"))
    else:
        for i in range(20):
            if i == (class_object-1):
                q_networks.append(get_q_network(weights_path + "/model" + str(i) + "h5"))
            else:
                q_networks.append(get_q_network("0"))
    return np.array([q_networks])