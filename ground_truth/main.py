import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter
import os

sns.set_theme()


def line_plot(data, title):
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=data)
    ax.set_title(title)
    plt.show()


def save_pickle_data(path, data):
    """Saves data in the pickle format
    :param path: Path to save
    :param data: Data to save
    :type path: str
    :type data: optional
    :return:
    """
    try:
        with open(path, "wb") as handler:
            pickle.dump(data, handler, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        if hasattr(e, "message"):
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
        with open(path, "rb") as handle:
            data = pickle.load(handle)
            return data
    except Exception as e:
        if hasattr(e, "message"):
            print(e.message)
            return None
        else:
            print(e)
            return None


critical_density = 26  # Density value above this are considered as congestion [veh/km/lane].
critical_speed = 65  # Speed values below this are considered as congestion [km/h].
n_segments = 160  # Number of freeway segments.
segment_ids = range(1, 161, 1)
segment_length = 50  # Freeway segment length [m].

data_dir = "./data/"
data_paths = []
interval_counter = 1

for file in os.listdir(data_dir):
    if file.endswith(".pkl") and "Link" in file:
        data_paths.append(os.path.join(data_dir, file))

data_paths.sort()
data_paths.sort(key=lambda x: len(x))

# A matrix with all bottleneck status data. Format: Time x Segment - cell 0 or 1 (1- bottleneck, 0- OK)
total_bottleneck_status = []

for path in data_paths:
    data_link = open_pickle(path)

    valid_data = dict(filter(lambda x: "E" not in x[0] and "gne" not in x[0], data_link.items()))
    ids = sorted(list(map(int, valid_data)))

    traff_params = []

    for link_id in ids:

        link_data = data_link[str(link_id)]

        df = pd.DataFrame(link_data, columns=["speed", "count"])

        counts = df.loc[df["count"] > 0, "count"]
        speeds = df.loc[df["count"] > 0, "speed"]

        # TODO: Check if this computation with 3 lanes is OK!!!!!!
        if 3400 <= link_id <= 3600 or 5450 <= link_id <= 5650:
            density = round(counts.sum() / counts.count() * 20 / 3, 2)
        else:
            density = round(counts.sum() / counts.count() * 20 / 2, 2)
        avg_speed = round(speeds.sum() / speeds.count() * 3.6, 2)
        flow = round(density * avg_speed, 2)

        traff_params.append((density, avg_speed, flow))

    traff_params = pd.DataFrame(np.array(traff_params), columns=["density", "avg_speed", "flow"])

    smoothed_dens = list(savgol_filter(traff_params["density"], 11, 3))
    smoothed_speed = list(savgol_filter(traff_params["avg_speed"], 11, 3))

    critical_d_ids_x = []
    critical_d_ids_y = []
    critical_s_ids_x = []
    critical_s_ids_y = []
    bottleneck_segments = []  # Ids of segments with bottleneck detected.
    bottleneck_status = []  # 0 or 1

    for i in range(0, len(smoothed_dens)):
        if smoothed_dens[i] >= critical_density:
            critical_d_ids_y.append(smoothed_dens[i])
            critical_d_ids_x.append(i + 1)
        if smoothed_speed[i] <= critical_speed:
            critical_s_ids_y.append(smoothed_speed[i])
            critical_s_ids_x.append(i + 1)
        if smoothed_dens[i] >= critical_density and smoothed_speed[i] <= critical_speed:
            bottleneck_segments.append(i + 1)
            bottleneck_status.append(1)
        else:
            bottleneck_status.append(0)

    # fig, ax = plt.subplots(1, 2)
    #
    # # For scatter plot (bottleneck visualization on graph).
    # x_bot = segment_ids
    # y_bot = [1] * 160
    #
    # # ax[0].plot(traff_params["density"], segment_ids)
    # ax[0].plot(segment_ids, smoothed_dens, color="black")
    # ax[0].plot(critical_d_ids_x, critical_d_ids_y, color="red")
    # ax[0].scatter(x_bot, y_bot, c=bottleneck_status, cmap="RdYlGn_r")
    # ax[0].hlines(y=critical_density, xmin=1, xmax=n_segments, linestyle='dashed', color="green")
    # ax[0].set_xlabel("Freeway segment")
    # ax[0].set_ylabel("Density (veh/km/lane)")
    #
    # # ax[1].plot(traff_params["avg_speed"])
    # ax[1].plot(segment_ids, smoothed_speed, color="black")
    # ax[1].plot(critical_s_ids_x, critical_s_ids_y, color="red")
    # ax[1].scatter(x_bot, y_bot, c=bottleneck_status, cmap="RdYlGn_r")
    # ax[1].hlines(y=critical_speed, xmin=1, xmax=n_segments, linestyle='dashed', color="green")
    # ax[1].set_xlabel("Freeway segment")
    # ax[1].set_ylabel("Speed (km/h)")
    #
    # fig.suptitle(f"Interval {interval_counter * 5}(s)")
    # plt.show()

    interval_counter += 1

    total_bottleneck_status.append(bottleneck_status)


total_bottleneck_status = np.array(total_bottleneck_status)

ax = plt.axes()
sns.heatmap(total_bottleneck_status, ax = ax)
ax.set_title('Ground truth bottleneck estimations')
ax.set_xlabel("Freeway segment")
ax.set_ylabel("Time (min)")
plt.show()

print()



