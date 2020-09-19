import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

from typing import List, Tuple


def from_array(array: np.ndarray, sample_rate: int = 1, filepath: str = "NPSound.wav"):
    """
    :param array: Array to convert into NPSound object
    :param sample_rate: Sample Rate of NPSound object
    :param filepath: Filepath of NPSound object
    :return: NPSound with desired parameters
    """
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

    def __init__(self, filepath: str = "NPSound.wav"):
        """
        :param filepath: Path to wav file
        """
        if filepath is not None:
            self.sample_rate, self.signal = wavfile.read(filepath)
        else:
            self.sample_rate, self.signal = 1, np.zeros((1,))
        self.filepath = filepath

    def copy(self):
        """
        :return: copy of this NPSound object
        """
        out = NPSound()
        out.filepath = self.filepath
        out.sample_rate = self.sample_rate
        out.signal = self.signal
        return out

    def plot(self, title: str = "np_sound:self.filepath", layered_plots: List = None,
             adjacent_plots: List = None, legend: List[str] = None):
        """
        :param title: Plot Supertitle
        :param layered_plots: Plots to layer on top of the NPSound's plot
        :param adjacent_plots: Plots to place next to the NPSound's plot
        :param legend: Legend for the NPSound's plot
        """
        fig, axs = plt.subplots(1 + (len(adjacent_plots) if adjacent_plots is not None else 0))
        fig.tight_layout(pad=3)
        base = axs[0] if adjacent_plots is not None and len(adjacent_plots) > 0 else axs

        base.plot(np.linspace(0, len(self.signal) / self.sample_rate, num=len(self.signal)), self.signal)

        true_title = self.filepath

        if layered_plots is not None and len(layered_plots) > 0:
            for plot in layered_plots:
                if isinstance(plot, NPSound):
                    if title == "np_sound:self.filepath":
                        true_title += ", " + plot.filepath
                    base.plot(np.linspace(0, len(plot.signal) / plot.sample_rate, num=len(plot.signal)), plot.signal)
                else:
                    base.plot(plot)

        base.set_title(true_title if title == "np_sound:self.filepath" else title)

        if adjacent_plots is not None and len(adjacent_plots) > 0:
            for x in range(1, len(adjacent_plots) + 1):
                i = x - 1
                if isinstance(adjacent_plots[i], NPSound):
                    if title == "np_sound:self.filepath":
                        axs[x].set_title(adjacent_plots[i].filepath)
                    axs[x].plot(np.linspace(0, len(adjacent_plots[i].signal) / adjacent_plots[i].sample_rate,
                                            num=len(adjacent_plots[i].signal)), adjacent_plots[i].signal)
                else:
                    axs[x].plot(adjacent_plots[i])

        if legend is not None and len(legend) > 0:
            plt.legend(legend, loc="upper right")

        plt.show()

    def _selection(self, start: float, end: float):
        """
        :param start: Starting point in seconds
        :param end: Ending point in seconds
        :return: Selection from [start] seconds to [end] seconds
        """
        if end == self.seconds_to_frames(-1):
            return self.signal[start:]
        return self.signal[start: end]

    def seconds_to_frames(self, seconds: float):
        """
        :param seconds: Seconds to be converted to frames
        :return: Frame equivalent of input seconds
        """
        return int(seconds * self.sample_rate)

    def _after_modify(self, out_signal: np.ndarray, new_filepath: str):
        """
        :param out_signal: Signal to process
        :param new_filepath: New filepath for output NPSound object
        :return: Processed NPSound object
        """
        if out_signal.shape[0] % 2 != 0:
            out_signal = np.pad(out_signal, (0, 1), 'edge')
        out_signal = out_signal.reshape((int(out_signal.size / 2), 2))
        return from_array(out_signal, sample_rate=self.sample_rate, filepath=new_filepath)

    def truncate_by_threshold(self, threshold: float, selection: Tuple[float, float] = None):
        """
        :param threshold: Audio threshold to truncate by
        :param selection: Only truncate this selection of audio
        :return: NPSound object with truncated signal
        """
        if selection is None:
            out_signal = self.signal[abs(self.signal) >= threshold]
            return self._after_modify(out_signal, self.filepath[:-4] + "(Threshold " + str(threshold) + ").wav")
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self._selection(selection[0], selection[1])
        modified_selection = selected_signal[abs(selected_signal) >= threshold]
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        return self._after_modify(out_signal, self.filepath[:-4] + "(Threshold " + str(threshold) + ").wav")

    def reverse(self, selection: Tuple[float, float] = None):
        """
        :param selection: Only truncate this selection of audio
        :return: NPSound object with reversed audio
        """
        if selection is None:
            out_signal = np.flip(self.signal)
            return self._after_modify(out_signal, self.filepath[:-4] + "(Reversed).wav")
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self._selection(selection[0], selection[1])
        modified_selection = np.flip(selected_signal)
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        return self._after_modify(out_signal, self.filepath[:-4] + "(Reversed).wav")

    def invert(self, selection: Tuple[float, float] = None):
        """
        :param selection: Only truncate this selection of audio
        :return: NPSound object with inverted audio
        """
        if selection is None:
            out_signal = -self.signal
            return self._after_modify(out_signal, self.filepath[:-4] + "(Inverted).wav")
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self._selection(selection[0], selection[1])
        modified_selection = -selected_signal
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        return self._after_modify(out_signal, self.filepath[:-4] + "(Inverted).wav")

    def shift_octaves(self, n: int):
        """
        :param n: Number of octaves to shift
        :return: Shifted audio
        """
        if n == 0:
            return self
        shift = 2.0 ** float(n)
        copy = self.copy()
        copy.filepath += "(Shifted {} octaves)".format(n)
        copy.sample_rate = int(copy.sample_rate * shift)
        return copy

    def amplify(self, percentage: float = 0, selection: Tuple[float, float] = None):
        """
        :param percentage: Amplify audio by this %
        :param selection: Only amplify this selection of audio
        :return: Amplified audio
        """
        if selection is None:
            out_signal = self.signal * (1 + (percentage / 100))
            return self._after_modify(out_signal, self.filepath[:-4] + "(Amplified {} percent).wav".format(percentage))
        selection = (self.seconds_to_frames(selection[0]), self.seconds_to_frames(selection[1]))
        selected_signal = self._selection(selection[0], selection[1])
        modified_selection = selected_signal * (1 + (percentage / 100))
        out_signal = join([self.signal[0: selection[0]],
                           modified_selection,
                           self.signal[selection[1]:] if selection[1] > 0 else np.array([])])
        return self._after_modify(out_signal, self.filepath[:-4] + "(Amplified {} percent).wav".format(percentage))

    def pad(self, seconds: Tuple[float, float]):
        """
        :param seconds: Tuple defining how much whitespace to append to (start, end) of sound file
        :return: Padded audio
        """
        frames = (self.seconds_to_frames(seconds[0]), self.seconds_to_frames(seconds[1]))

        start_padding = np.zeros((frames[0], 2))
        end_padding = np.zeros((frames[1], 2))

        out = from_array(start_padding, sample_rate=self.sample_rate)
        out += self
        out += from_array(end_padding, sample_rate=self.sample_rate)

        out.filepath = self.filepath[:-4] + "(Padded ({}, {})).wav".format(seconds[0], seconds[1])

        return out

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

    def __add__(self, other):
        out = self.copy()
        if isinstance(other, np.ndarray):
            if out.signal.shape[1] != other.shape[1]:
                raise Exception("Dimensionality of np.ndarray does not match the signal. Signal Shape {}, np.ndarray "
                                "Shape {}".format(self.signal.shape, other.shape))
            out.signal = np.concatenate((self.signal, other), axis=0)
        elif isinstance(other, NPSound):
            if len(out.signal.shape) + len(other.signal.shape) < 2 and out.signal.shape[1] != other.signal.shape[1]:
                raise RuntimeError("Dimensionality of NPSound does not match the signal. Signal Shape {}, NPSound "
                                   "Shape {}".format(self.signal.shape, other.signal.shape))
            if out.sample_rate != other.sample_rate:
                raise RuntimeError("Sample rates do not match. {}, {}".format(self.sample_rate, other.sample_rate))
            out.signal = np.concatenate((self.signal, other.signal), axis=0)
            out.filepath = out.filepath[:-4] + "_" + other.filepath
        return out

    def __mul__(self, other):
        if not isinstance(other, int):
            raise RuntimeError("Cannot multiply type NPSound by type {}".format(str(type(other))))
        out = self.copy()
        temp_fp = out.filepath

        for i in range(other):
            out += self.copy()

        out.filepath = temp_fp[:-4] + "x{}".format(other) + ".wav"
        return out

    def __len__(self):
        return int(self.len())

    def __str__(self):
        return "Filepath: {}\nLength: {}s\nSample Rate: {}\nSignal Shape: {}\n".format(self.filepath, self.len(),
                                                                                       self.sample_rate,
                                                                                       self.signal.shape)
