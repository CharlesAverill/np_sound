import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.io.wavfile import read
from typing import List

arrs = []

for root, dirs, files in os.walk("."):
    for x in files:
        print(root, x)
        if x.endswith(".wav"):
            wav_file = read(os.path.join(root, x))
            wav_arr = np.array(wav_file[1], dtype=float).flatten()
            arrs.append((os.path.join(root, x), wav_arr))

for f in arrs:
    """
    plt.title(file[0])
    plt.plot(file[1])
    plt.plot(file[1][abs(file[1]) > 50])
    plt.legend(["Baseline", "Truncated"])
    plt.show()
    """


class NPSound:

    def __init__(self, filepath: str = None, flatten: bool = True):
        file = read(filepath)
        self.filepath = filepath
        self.ndarray = np.array(file[1], dtype=float).flatten() if flatten else np.array(file[1], dtype=float)

    def plot(self, title: str = "np-sound:self.filepath", concurrent_plots: List[np.ndarray] = None,
             adjacent_plots: List[np.ndarray] = None, legend: List[str] = None):
        """
        :param title: Plot Supertitle
        :param concurrent_plots: Plots to layer on top of the NDSound's plot
        :param adjacent_plots: Plots to place next to the NDSound's plot
        :param legend: Legend for the NDSound's plot
        :return:
        """
        fig, axs = plt.subplots(1 + (len(adjacent_plots) if adjacent_plots is not None else 0))
        base = axs[0] if adjacent_plots is not None and len(adjacent_plots) > 0 else axs

        fig.suptitle(title if title != "np-sound:self.filepath" else self.filepath)

        base.plot(self.ndarray)
        if concurrent_plots is not None and len(concurrent_plots) > 0:
            [base.plot(plot) for plot in concurrent_plots]

        if adjacent_plots is not None and len(adjacent_plots) > 0:
            for i in range(len(adjacent_plots)):
                axs[i + 1].plot(adjacent_plots[i])

        if legend is not None and len(legend) > 0:
            plt.legend(legend)

        plt.show()


sound = NPSound("out000.wav")
sound.plot()