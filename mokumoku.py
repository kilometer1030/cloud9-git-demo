# coding: utf-8
from sys import argv as arg
import wave

import numpy as np
from numpy import sqrt, mean, square
import scipy.signal as sig
import matplotlib.pyplot as plt


class Wav:
    """
    Attributes
    ----------
    is_stereo : bool
        ``True`` if stereo, ``False`` otherwise.
    rate : int
        Sampling rate.
    length : int
        Number of audio frames.
    depth : int
        Bit depth.
    data : bytes
        Frames of audio.
    datatype : dict
        Dictionary mapping `depth` to ``dtype`` objects.
    """
    
    datatype = {16: np.int16, 32: np.int32}
    
    def __init__(self, file):
        """
        Parameters
        ----------
        file : str
            The WAV filename (e.g. 'foo.wav').
        """
        
        wf = wave.open(file, 'rb')
        
        self.is_stereo = wf.getnchannels() == 2
        self.rate = wf.getframerate()
        self.length = wf.getnframes()
        self.depth = 8 * wf.getsampwidth()
        self.data = wf.readframes(-1)
        
        wf.close()


def analyze(file, Y=240, N=441):
    """Analyze a WAV file to figure the time variation of BPM.
    
    Parameters
    ----------
    file : str
        The WAV filename (e.g. 'foo.wav').
    Y : int, optional
        Estimated BPM to set the y-range of the plot.
    N : int, optional
        Frame size used when calculating RMS.
    """
    
    # get attributes of the WAV file
    wav = Wav(file)
    
    # binary -> integer
    data = np.frombuffer(wav.data, dtype=wav.datatype[wav.depth])
    
    # integer -> float (normalize)
    data = data / 2**(wav.depth - 1)
    
    if wav.is_stereo:
        l_ch = data[::2]
        r_ch = data[1::2]
        data = (l_ch + r_ch) / 2
    
    # 1-D array -> ?-by-N matrix w/ zero padding (as needed)
    data = np.pad(data, (0, -wav.length % N), 'constant')\
             .reshape((-1, N))
    
    # calculate RMS for a set of N samples
    rms = sqrt(mean(square(data), axis=1))
    
    # let `beat` be the increase in RMS
    beat = np.gradient(rms)
    beat[beat < 0] = 0
    
    # FFT
    f, t, Sxx = sig.spectrogram(beat,
                                fs=60*wav.rate/N,
                                window='boxcar',
                                nperseg=3000,
                                noverlap=2900,
                                )
    
    # plot the spectrogram
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(t, f, Sxx)
    plt.title(file)
    plt.xlabel('Time [min]')
    plt.ylabel('BPM')
    plt.ylim(Y*2/3, Y*4/3)
    plt.savefig('output.png')


if __name__ == '__main__':
    if len(arg) < 2:
        print('Usage: $ python', arg[0], 'foo.wav [estimated_BPM [frame_size]]')
    else:
        arg[2:] = [int(i) for i in arg[2:]]
        analyze(*arg[1:])