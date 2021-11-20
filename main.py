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
import ground_truth

from itertools import islice
import numpy as np
import os


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
        if n_vehicles == "all":
            return data
        else:
            return dict(islice(data.items(), int(n_vehicles)))
    except ValueError:
        raise ValueError("Method get_data() takes string 'all' or integer.")


def change_length(traff_data, length: int):
    """Change link length

    Changes the length of the observed highway links. Useful when original links are
    short (ex. 50m) and you want to run a this methodology on larger links (ex. 500m)

    :param traff_data: Routes data
    :param length: Highway link length
    :type traff_data: dict
    :type length: int
    :return:
    """
    new_lengths = list(range(0, 8001, length))

    for route_id in traff_data:

        # If there is only one entry, skip
        if len(traff_data[route_id]) == 1:
            continue

        for record_id in range(0, len(traff_data[route_id])):
            _id = traff_data[route_id][record_id][1]

            # Link with id E0 is the input link for simulation
            if _id == "E0":
                continue

            for i in range(0, len(new_lengths) - 1):
                first = new_lengths[i]
                second = new_lengths[i + 1]

                if first < int(_id) <= second:
                    _id = second
                    traff_data[route_id][record_id][1] = str(second)


def main():

    data_dir = "/home/leo/PycharmProjects/highwayBottleneck/data/"
    data_paths = []
    matrix_bprob_proposed = []

    interval_counter = 1

    # Get all data from 24 time intervals.
    for file in os.listdir(data_dir):
        if file.endswith(".pkl") and "Veh" in file:
            data_paths.append(os.path.join(data_dir, file))

    for dp in data_paths:

        # Gets routes data.
        # routes_data = get_data(r"./data/all_data/TrafData.pkl", "all")
        routes_data = get_data(dp, "all")

        # Use only if needed
        # change_length(routes_data, 500):

        # Generates transitions from routes data.
        transitions = STMops.make_transitions(routes_data)

        # Initializes the FIS.
        fuzzy = FuzzyBott()

        # Plot input and output variables from FIS
        fuzzy.plot_vars()

        bot_prob_edges = []

        for trans in transitions.keys():
            speeds = np.array(transitions[trans])

            # Computes the STM.
            stm = STMops.compute_stm(speeds[:, 0], speeds[:, 1])

            # dd - diagonal distance, sd - source distance
            dd, sd = STMops().get_distances(stm)

            # Result of the FIS.
            result = fuzzy.get_bot_prob(diag_dist=dd, origin_dist=sd)

            bot_prob_edges.append(result)

            # Plots STM with corresponding highway bottleneck probability.
            # plot_heatmap(stm, "{0}; {1}".format(trans, result))

        matrix_bprob_proposed.append(bot_prob_edges)

    print()

    # TODO: Pass the matrix_bprob_proposed to plot_eval_data() and plot it.
    # ground_truth.plot_eval_data()


if __name__ == "__main__":
    main()
