"""Fuzzy highway bottleneck estimation using speed transition matrix

This code illustrates the usage of the Fuzzy Inference System (FIS)
for estimation of the highway bottlenecks using the Speed Transition
Matrices (STMs).
More about STMs:
https://medium.com/analytics-vidhya/speed-transition-matrix-novel-road-traffic-data-modeling-technique-d37bd82398d1
"""

from misc import config as cfg

__licence__ = cfg.LICENCE
__author__ = cfg.AUTHOR
__email__ = cfg.EMAIL
__status__ = cfg.STATUS
__docformat__ = cfg.DOCFORMAT


from misc.misc import open_pickle, plot_heatmap
from FuzzySTM import FuzzyBott, STMops
from itertools import islice
import numpy as np


def get_data(file_name, n_vehicles):
    """Gets traffic data saved as pickle.

    Traffic data is saved as dictionary of routes.
    'route_name': route
    Every route is a list that consists of list with two entries: [speed [mph], 'road_segment'].
    ex.
    route[0] = [19.012, 'E1']
    route[1] = [23.611, 'E1']
    route[2] = [22.536, 'E2']
    One 'route' represents the movement of one vehicle through the observed highway.

    :param n_vehicles: String 'all' orn number of vehicles to observe.
    :param file_name: Name or path to the file with traffic data.
    :type n_vehicles: str, optional
    :type file_name: str
    :return: Returns dictionary of routes.
    :rtype: dict
    """
    data = open_pickle(file_name)
    try:
        if n_vehicles == 'all':
            return data
        else:
            return dict(islice(data.items(), int(n_vehicles)))
    except ValueError:
        raise ValueError("Method get_data() takes string 'all' or integer.")


def main():
    # Gets routes data.
    routes_data = get_data('TrafData.pkl', 100)

    # Generates transitions from routes data.
    transitions = STMops.make_transitions(routes_data)

    # Initializes the FIS.
    fuzzy = FuzzyBott()

    # Plot input and output variables from FIS
    fuzzy.plot_vars()

    for trans in transitions.keys():
        speeds = np.array(transitions[trans])

        # Computes the STM.
        stm = STMops.compute_stm(speeds[:, 0], speeds[:, 1])

        # dd - diagonal distance, sd - source distance
        dd, sd = STMops().get_distances(stm)

        # Result of the FIS.
        result = fuzzy.get_bot_prob(diag_dist=dd, origin_dist=sd)

        # Plots STM with corresponding highway bottleneck probability.
        plot_heatmap(stm, '{0}; {1}'.format(trans, result))


if __name__ == '__main__':
    main()
