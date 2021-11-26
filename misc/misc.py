"""Miscellaneous functions

This file contains all kinds of useful functions:
- save_pickle_data - Saves data in the pickle format
- open_pickle - Opens pickle data from defined path
- rtm - Function for rounding integer value to higher / lower value based on multiple value
- plot_heatmap - Plots heatmap for STM
"""

from misc import config as cfg

__licence__ = cfg.LICENCE
__author__ = cfg.AUTHOR
__email__ = cfg.EMAIL
__status__ = cfg.STATUS
__docformat__ = cfg.DOCFORMAT


import pickle
from misc import config
import numpy as np
import matplotlib.pyplot as plt
import os


def get_data_paths(data_dir: str, data_type: str):
    """
    Get all data from 24 time intervals.
    Data is saved into pickles, one pickle for 5 min intervals.
    :param data_dir: Path to directory with data.
    :param data_type: Type of data -> Veh or Link.
    :type data_dir: str
    :type data_type: str
    :return:
    """
    data_paths = []

    for file in os.listdir(data_dir):
        if file.endswith(".pkl") and data_type in file:
            data_paths.append(os.path.join(data_dir, file))

    data_paths.sort()
    data_paths.sort(key=lambda x: len(x))

    return data_paths


def save_pickle_data(path, data):
    """Saves data in the pickle format

    :param path: Path to save
    :param data: Data to save
    :type path: str
    :type data: optional
    :return:
    """
    try:
        with open(path, 'wb') as handler:
            pickle.dump(data, handler, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)


def open_pickle(path):
    """Opens pickle data from defined path

    :param path: Path to pickle file
    :type path: str
    :return:
    """
    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            return data
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return None
        else:
            print(e)
            return None


def rtm(x, base=5):
    """Function for rounding integer value to higher / lower value based on multiple value

    :param x: Number for rounding
    :param base: Multiple that will represent the number value
    :type x: int
    :type base: int
    :return: Rounded number
    :rtype: int
    """
    if 100 < x <= 110:
        return 100
    if x > 110:
        return None
    return int(base * round(x / base))


def plot_heatmap(data, title, output='show', filename='image.png'):
    """Plots heatmap for STM

    :param data: Speed transition matrix to plot
    :param title: Title of the plot
    :param output: Show or save an image - input must be 'show' or 'save'
    :param filename: Name if image must be saved
    :type data: ndarray
    :type title: str
    :type output: str
    :type filename: str
    :return: Show or save the plot
    """
    states_names = config.SPEED_LIST

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='cividis', interpolation='none')

    # cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel('Number of vehicles')

    ax.set_xticks(np.arange(len(states_names)))
    ax.set_yticks(np.arange(len(states_names)))
    ax.set_xticklabels(states_names)
    ax.set_yticklabels(states_names)

    plt.xlabel('Destination speed (%)')
    plt.ylabel('Origin speed (%)')

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # for i in range(len(states_names)):
    #     for j in range(len(states_names)):
    #         ax.text(j, i, data[i, j], ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()

    if output == 'show':
        plt.grid(True)
        plt.show()
    if output == 'save':
        plt.savefig(filename, bbox_inches='tight')
