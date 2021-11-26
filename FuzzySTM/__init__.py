"""Implementation of the FIS and other classes

This file contains two classes:
- FuzzyBott: Class for initialization and evaluation of the fuzzy system.
- STMops: Class for speed transition matrix (STM) analysis and computation.
"""

from misc import config as cfg

__licence__ = cfg.LICENCE
__author__ = cfg.AUTHOR
__email__ = cfg.EMAIL
__status__ = cfg.STATUS
__docformat__ = cfg.DOCFORMAT

from simpful import *
import matplotlib.pyplot as plt
from scipy.spatial import distance
from misc import config
from misc.misc import rtm
from math import sqrt
import numpy as np
from collections import OrderedDict


class FuzzyBott(object):
    """Class for initialization and evaluation of the fuzzy system

    **Methods**
    - plot_vars: Method for plotting input and output variables
    - get_bot_prob: Method for calculating the bottleneck probability
    """

    def __init__(self):
        """Constructor will initialize Fuzzy system with all corresponding
        input and output variable and rules."""
        self.fs = FuzzySystem()

        # Input variable distance from diagonal.
        p1 = FuzzySet(function=InvSigmoid_MF(c=0.25, a=20), term="small")
        p2 = FuzzySet(function=Gaussian_MF(mu=0.5, sigma=0.1), term="medium")
        p3 = FuzzySet(function=Sigmoid_MF(c=0.75, a=20), term="large")
        self.input_1 = LinguisticVariable(
            [p1, p2, p3],
            concept="Distance from diagonal",
            universe_of_discourse=[0, 1],
        )
        self.fs.add_linguistic_variable("diag_dist", self.input_1)

        # Input variable distance from origin.
        p1 = FuzzySet(function=InvSigmoid_MF(c=0.25, a=20), term="small")
        p2 = FuzzySet(function=Gaussian_MF(mu=0.5, sigma=0.1), term="medium")
        p3 = FuzzySet(function=Sigmoid_MF(c=0.75, a=20), term="large")
        self.input_2 = LinguisticVariable(
            [p1, p2, p3],
            concept="Distance from origin",
            universe_of_discourse=[0, 1],
        )
        self.fs.add_linguistic_variable("origin_dist", self.input_2)

        # Output variable bottleneck probability.
        p1 = FuzzySet(function=InvSigmoid_MF(c=0.2, a=20), term="small")
        p2 = FuzzySet(function=Gaussian_MF(mu=0.4, sigma=0.1), term="medium")
        # p3 = FuzzySet(function=Sigmoid_MF(c=0.6, a=20), term="large")
        p3 = FuzzySet(function=Sigmoid_MF(c=0.55, a=20), term="large")
        self.output_1 = LinguisticVariable(
            [p1, p2, p3],
            concept="Bottleneck probability",
            universe_of_discourse=[0, 1],
        )
        self.fs.add_linguistic_variable("bot_prob", self.output_1)

        # Linguistic variables for output.
        self.fs.set_crisp_output_value("small", 0.0)
        self.fs.set_crisp_output_value("medium", 0.5)
        self.fs.set_crisp_output_value("large", 1.0)

        # Set of rules.
        rules = []
        rule_set = [
            "IF (diag_dist IS small) AND (origin_dist IS small) THEN (bot_prob IS large)",
            "IF (diag_dist IS small) AND (origin_dist IS medium) THEN (bot_prob IS medium)",
            "IF (diag_dist IS small) AND (origin_dist IS large) THEN (bot_prob IS small)",
            "IF (diag_dist IS medium) AND (origin_dist IS small) THEN (bot_prob IS medium)",
            "IF (diag_dist IS medium) AND (origin_dist IS medium) THEN (bot_prob IS medium)",
            "IF (diag_dist IS medium) AND (origin_dist IS large) THEN (bot_prob IS small)",
            "IF (diag_dist IS large) AND (origin_dist IS small) THEN (bot_prob IS large)",
            "IF (diag_dist IS large) AND (origin_dist IS medium) THEN (bot_prob IS medium)",
            "IF (diag_dist IS large) AND (origin_dist IS large) THEN (bot_prob IS large)",
        ]
        for rule in rule_set:
            rules.append(rule)
        self.fs.add_rules(rules)

    def plot_vars(self):
        """Method for plotting input and output variables

        :return: Shows the plot
        """
        fig, ax = plt.subplots(3, 1, figsize=(7, 5))
        self.input_1.draw(ax=ax[0])
        self.input_2.draw(ax=ax[1])
        self.output_1.draw(ax=ax[2])
        plt.show()

    def get_bot_prob(self, diag_dist: float, origin_dist: float) -> float:
        """Method for calculating the bottleneck probability

        :param diag_dist: Distance of the center of mass to the diagonal
        :param origin_dist: Distance of the center of mass to the origin (0,0)
        :type diag_dist: float
        :type origin_dist: float
        :return: Returns bottleneck probability
        :rtype: float
        """
        self.fs.set_variable("diag_dist", diag_dist)
        self.fs.set_variable("origin_dist", origin_dist)
        return round(
            self.fs.Sugeno_inference(["bot_prob"], verbose=False)["bot_prob"],
            4,
        )


class STMops(object):
    """Class for speed transition matrix (STM) analysis and computation

    **Methods**
    - get_mass_center - Computes the center of mas of the STM
    - diag_dist - Computes distance from center of mass to diagonal
    - from_origin_distance - Computes distance from center of mass to origin of the STM (0,0)
    - get_distances - Computes distances from center of mass to: 1) diagonal, 2) STM origin
    - get_harmonic_speed - Computes relative harmonic mean speed of provided speeds
    - make_transitions - Creates transitions from traffic data necessary to compute STMs
    - compute_stm - Computes the STM
    """

    @staticmethod
    def get_mass_center(m):
        """Computes the center of mas of the STM

        :param m: Speed transition matrix
        :type m: ndarray
        :return: Center of mass (x, y)
        :rtype: tuple
        """
        max_val = 0.2 * np.max(m)  # Filter: remove 20% of maximal value.
        m = np.where(m < max_val, 0, m)
        m = m / np.sum(m)
        # marginal distributions
        dx = np.sum(m, 1)
        dy = np.sum(m, 0)
        # expected values
        X, Y = m.shape
        cx = np.sum(dx * np.arange(X))
        cy = np.sum(dy * np.arange(Y))
        return (int(cx), int(cy))

    @staticmethod
    def diag_dist(point):
        """Computes distance from center of mass to diagonal

        :param point: Center of mass
        :type point: tuple
        :return: Euclidean distance from center of mass to diagonal
        :rtype: float
        """
        # Max distance to the diagonal (square matrix m x m) is: diagonal_length / 2.
        max_d = (config.MAX_INDEX * sqrt(2)) / 2
        distan = []
        for d in config.DIAG_LOCS:
            distan.append(distance.euclidean(d, point))
        return round(min(distan) / max_d, 2)  # Relative distance.

    @staticmethod
    def from_origin_distance(point):
        """Computes distance from center of mass to origin of the STM (0,0)

        :param point: Center of mass
        :type point: tuple
        :return: Euclidean distance from center of mass to origin of the STM
        :rtype: float
        """
        max_point = (config.MAX_INDEX, config.MAX_INDEX)
        origin = (0, 0)
        max_d = distance.euclidean(origin, max_point)
        return round(distance.euclidean(origin, point) / max_d, 2)

    @staticmethod
    def get_distances(stm):
        """Computes distances from center of mass to: 1) diagonal, 2) STM origin

        :param stm: Speed transition matrix
        :type stm: ndarray
        :return: Distances from center of mass to 1) diagonal, 2) STM origin
        :rtype: float, float
        """
        point = STMops.get_mass_center(stm)
        return STMops.diag_dist(point), STMops.from_origin_distance(point)

    @staticmethod
    def get_harmonic_speed(speed_data, limit=130):
        """Computes relative harmonic mean speed of provided speeds

        :param speed_data: List of speeds.
        :param limit: Speed limit on the observed highway
        :type speed_data: list
        :type limit: int
        :return: Relative harmonic mean speed of provided speeds (0-100%)
        :rtype: int
        """
        hSum = 0
        for s in speed_data:
            if s == 0:
                hSum += 1
                continue
            hSum += 1 / s
        # 3.6 - conversion to kmh, limit * 100 - conversion to relative speed
        result = round(len(speed_data) / hSum * 3.6 / limit * 100, 4)
        # If relative speed is more than 100%
        if result > 100:
            return 100
        return result

    @staticmethod
    def make_transitions(data):
        """Creates transitions from traffic data necessary to compute STMs

        Traffic data example:
        Traffic data is saved as dictionary of routes.
        'route_name': route
        Every route is a list that consists of list with two entries: [speed [mph], 'road_segment'].
        ex.
        route[0] = [19.012, 'E1']
        route[1] = [23.611, 'E1']
        route[2] = [22.536, 'E2']
        One 'route' represents the movement of one vehicle through the observed highway.

        :param data: Traffic data -routes from highway.
        :type data: dict
        :return: Dictionary of transitions
        :rtype: dict
        """
        transitions = {}
        for veh_id in data.keys():
            veh_data = data[veh_id]

            road_id_list = []
            for v in veh_data:
                if v[1] not in road_id_list:
                    road_id_list.append(v[1])

            for i in range(0, len(road_id_list) - 1):

                from_id = road_id_list[i]
                to_id = road_id_list[i + 1]

                # Difference between ids must be 50 (m).
                # If difference is 100, data is missing.
                if from_id != "E0":
                    if (int(to_id) - int(from_id)) >= 100:
                        continue

                # Get origin (from_data) and destination (to_data) speeds
                from_data = np.array(
                    list(filter(lambda x: x[1] == str(from_id), veh_data))
                )
                to_data = np.array(
                    list(filter(lambda x: x[1] == str(to_id), veh_data))
                )

                from_speed = STMops.get_harmonic_speed(
                    from_data[:, 0].astype("float")
                )
                to_speed = STMops.get_harmonic_speed(
                    to_data[:, 0].astype("float")
                )

                if from_id == "E0":
                    from_id = 0
                    to_id = int(to_id)
                else:
                    from_id = int(from_id)
                    to_id = int(to_id)

                # Name of the transition OriginIDtoDestinationID
                trans_id = (from_id, to_id)

                if trans_id not in transitions.keys():
                    transitions[trans_id] = []

                transitions[trans_id].append([from_speed, to_speed])

        return OrderedDict(sorted(transitions.items()))

    @staticmethod
    def compute_stm(origin_speeds, dest_speeds):
        """Computes the STM

        :param origin_speeds: List of origin relative speeds
        :param dest_speeds: List of destination relative speeds
        :type origin_speeds: ndarray
        :type dest_speeds: ndarray
        :return: Speed transition matrix
        :rtype: ndarray
        """
        resolution, max_index = config.RESOLUTION, config.MAX_INDEX
        t_matrix = np.zeros((max_index, max_index))

        if len(origin_speeds) > 0 and len(dest_speeds) > 0:
            for i in range(0, len(origin_speeds)):

                ############################################
                # (absolute) If the speed is larger than 100 and less than 140
                # (relative) If the relative speed is larger than 110%
                if origin_speeds[i] == None or dest_speeds[i] == None:
                    continue
                ###############################################

                c_route_speed_index = int(
                    rtm(origin_speeds[i]) / resolution - 1
                )
                n_route_speed_index = int(rtm(dest_speeds[i]) / resolution - 1)

                t_matrix[c_route_speed_index, n_route_speed_index] += 1
            return t_matrix.astype("int")
        else:
            return t_matrix.astype("int")
