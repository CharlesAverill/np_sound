import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

from typing import List, Tuple


def from_array(array: np.ndarray, sample_rate: int = 1, filepath: str = "NPSound.wav"):
    out = NPSound()
    out.signal = array
    out.sample_rate = sample_rate
    out.filepath = filepath
    return out


def join(signals: List[np.ndarray]):
    """
    :param signals: List of signals to join together
    :return: An np.ndarray of joined signals
    """
    out = signals[0]
    for signal in signals[1:]:
        out = np.concatenate((out, signal), axis=0)
    return out


class NPSound:

    def __init__(self, filepath: str = None):
        """
        :param filepath: Path to wav file
        """
        if filepath is not None:
            self.sample_rate, self.signal = wavfile.read(filepath)
        else:
            self.sample_rate, self.signal = 1, np.zeros((1,))
        self.filepath = filepath

    def plot(self, title: str = "np-sound:self.filepath", layerd_plots: List = None,
             adjacent_plots: List = None, legend: List[str] = None):
        """
        :param title: Plot Supertitle
        :param layerd_plots: Plots to layer on top of the NDSound's plot
        :param adjacent_plots: Plots to place next to the NDSound's plot
        :param legend: Legend for the NDSound's plot
        :return:
        """
        fig, axs = plt.subplots(1 + (len(adjacent_plots) if adjacent_plots is not None else 0))
        fig.tight_layout(pad=3)
        base = axs[0] if adjacent_plots is not None and len(adjacent_plots) > 0 else axs

        base.plot(np.linspace(0, len(self.signal) / self.sample_rate, num=len(self.signal)), self.signal)

        true_title = self.filepath

        if layerd_plots is not None and len(layerd_plots) > 0:
            for plot in layerd_plots:
                if isinstance(plot, NPSound):
                    if title == "np-sound:self.filepath":
                        true_title += ", " + plot.filepath
                    base.plot(np.linspace(0, len(plot.signal) / plot.sample_rate, num=len(plot.signal)), plot.signal)
                else:
                    base.plot(plot)

        base.set_title(true_title if title == "np-sound:self.filepath" else title)

        if adjacent_plots is not None and len(adjacent_plots) > 0:
            for x in range(1, len(adjacent_plots) + 1):
                i = x - 1
                if isinstance(adjacent_plots[i], NPSound):
                    if title == "np-sound:self.filepath":
                        axs[x].set_title(adjacent_plots[i].filepath)
                    axs[x].plot(np.linspace(0, len(adjacent_plots[i].signal) / adjacent_plots[i].sample_rate,
                                            num=len(adjacent_plots[i].signal)), adjacent_plots[i].signal)
                else:
                    axs[x].plot(adjacent_plots[i])

        if legend is not None and len(legend) > 0:
            plt.legend(legend, loc="upper right")

        plt.show()

    def selection(self, start: int, end: int):
        """
        :param start: Starting point in seconds
        :param end: Ending point in seconds
        :return: Selection from [start] seconds to [end] seconds
        """
        if end == self.seconds_to_frames(-1):
            return self.signal[start:]
        return self.signal[start: end]

    def seconds_to_frames(self, seconds: int):
        """
        :param seconds: Seconds to be converted to frames
        :return: Frame equivalent of input seconds
        """
        return seconds * self.sample_rate

    def truncate_by_threshold(self, threshold: float, selection: Tuple[int, int] = None):
        """
        :param threshold: Audio threshold to truncate by
        :param selection: Only truncate this selection of audio
        :return: NPSound object with truncated signal
        """
        if selection is None:
            return from_array(self.signal[abs(self.signal) >= threshold], sample_rate=self.sample_rate,
                              filepath=self.filepath[:-4] + "(Threshold " + str(threshold) + ").wav")
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self.selection(selection[0], selection[1])
        modified_selection = selected_signal[abs(selected_signal) >= threshold]
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        out_signal = out_signal.reshape((int(out_signal.size / 2), 2))
        return from_array(out_signal, sample_rate=self.sample_rate,
                          filepath=self.filepath[:-4] + "(Threshold " + str(threshold) + ").wav")

    def reverse(self, selection: Tuple[int, int] = None):
        """
        :param selection: Only truncate this selection of audio
        :return: NPSound object with reversed audio
        """
        if selection is None:
            return from_array(np.flip(self.signal), sample_rate=self.sample_rate,
                              filepath=self.filepath[:-4] + "(Reversed).wav")
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self.selection(selection[0], selection[1])
        modified_selection = np.flip(selected_signal)
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        out_signal = out_signal.reshape((int(out_signal.size / 2), 2))
        return from_array(out_signal, sample_rate=self.sample_rate, filepath=self.filepath[:-4] + "(Reversed).wav")

    def invert(self, selection: Tuple[int, int] = None):
        """
        :param selection: Only truncate this selection of audio
        :return: NPSound object with inverted audio
        """
        if selection is None:
            return from_array(-self.signal, sample_rate=self.sample_rate,
                              filepath=self.filepath[:-4] + "(Inverted).wav")
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self.selection(selection[0], selection[1])
        modified_selection = -selected_signal
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        out_signal = out_signal.reshape((int(out_signal.size / 2), 2))
        return from_array(out_signal, sample_rate=self.sample_rate, filepath=self.filepath[:-4] + "(Inverted).wav")

    def write(self, filepath: str = None):
        """
        :param filepath: Filepath to write to if the default filepath does not suffice
        :return: T/F if the write was successful
        """
        try:
            wavfile.write(filepath if filepath is not None else self.filepath, self.sample_rate, self.signal)
            return True
        except Exception as e:
            print(e)
            return False

    def len(self):
        return len(self.signal) / self.sample_rate

    def __len__(self):
        return int(self.len())

    def __str__(self):
        return "Filepath: {}\nLength: {}s\nSample Rate: {}\nSignal Shape: {}\n".format(self.filepath, self.len(),
                                                                                       self.sample_rate,
                                                                                       self.signal.shape)
