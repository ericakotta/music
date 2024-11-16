import scipy
import matplotlib.pyplot as plt
import numpy as np
import tools
import scipy
import time
from key_signature_analysis import get_key_signature
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import argparse

def parse_args(args):

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--wav-file',
        type=str,
        default="Super Mario Bros. Theme  but its in a minor key.wav",
        help='Name of .wav file (in current directory) to analyze key signature. If not specified, searches directory and uses first .wav file found.'
    )
    arg_parser.add_argument(
        '--conv-width',
        type=float,
        default=0.05,
        help='Width to convolute '
    )
    arg_parser.add_argument(
        '--start-seconds',
        type=float,
        default=0.0,
        help='Sample start time. Defaults to start of wav file.',
    )
    arg_parser.add_argument(
        '--end-seconds',
        type=float,
        default=None,
        help='Sample end time. Defaults to end of wav file.'
    )

    return arg_parser.parse_args()

# Function to handle SIGINT (Ctrl+C)
def handle_interrupt(signal, frame):
    print("Ctrl + C pressed. Closing the figure.")
    plt.close('all')  # Close all open figure windows

# Register the signal handler for SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, handle_interrupt)


def get_fft(x, y):
    '''Return x_fft and y_fft'''
    N = len(y)
    dx = np.mean(np.abs(x[1:] - x[:-1]))
    yf = scipy.fft.fft(y)
    xf = scipy.fft.fftfreq(N, dx)[:N//2]
    # Sort xf
    idx = xf.argsort()
    xf, yf = xf[idx], yf[idx]

    return xf, yf


def get_fft_(x, y, comp=''):
    '''Get specified component of fft in fft-x order'''
    dx = np.abs(np.mean([x[i+1] - x[i] for i in range(len(x)-1)]))
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(len(y), d=dx)
    idx = np.argsort(xf)
    if comp=='abs':
        yf = np.abs(yf)
    elif comp == 'real':
        yf = yf.real
    elif comp == 'imag':
        yf = yf.imag
    return (xf[idx], yf[idx])

def convolve_with_gaussian(x, y, sigma):
    '''Broaden a curve using gaussian of width sigma, in  units of x
    Make sure x is ordered!'''
    dx = np.mean([np.abs(x[i+1] - x[i]) for i in range(len(x)-2)])
    # print(f"dx: {dx}")
    gx = np.arange(-3 * sigma, 3 * sigma, dx)
    # print("Len gx:",len(gx))
    g = np.exp(-(gx / sigma) ** 2 / 2)
    # plt.plot(gx, gaussian

    n = len(x) + len(gx) - 1
    y_padded = np.pad(y, (0, n - len(y)), mode='constant')
    g_padded = np.pad(g, (0, n - len(g)), mode='constant')
    y_conv = np.fft.ifft(
        np.fft.fft(y_padded) * np.fft.fft(g_padded)
    )
    y_conv = np.real(y_conv)
    y_conv = y_conv[:len(y)]
    # y_conv = np.convolve(y, g, mode="same")
    # y_conv = (y_conv - min(y_conv)) / (max(y_conv) - min(y_conv))
    return y_conv
    # return convolve_1d(y, gaussian)

def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    return [
        np.dot(
            signal[max(0,i):min(i+len(kernel),len(signal))],
            kernel[max(-i,0):len(signal)-i*(len(signal)-len(kernel)<i)],
        )
        for i in range(1-len(kernel),len(signal))
    ]


def get_gaussian_sample(x, y, x0, sigma, crop_sigma=None):
    '''Return element-wise multiplication of your sample data with a gaussian of width sigma centered at x0'''
    if crop_sigma:
        idx = np.where(np.abs(x - x0) <= crop_sigma * sigma)
        gaussian_x = x[idx]
        gaussian_y = y[idx]
    else:
        idx = np.where(np.abs(x) >= 0)
        gaussian_x = x.copy()
        gaussian_y = y.copy()
    gaussian = np.exp(-((gaussian_x - x0)/sigma) ** 2 / 2)
    return idx, np.multiply(gaussian_y, gaussian)



class musicTranscriber():
    """
    Class to input .wav piano audio data and transcribe a pdf of sheet music
    """
    def __init__(self, input_wav_filename, display=False):
        # self.verbose = verbose
        self.display = display
        if self.display:
            # Get a colormap for 88 keys
            self.cmap = plt.get_cmap('gnuplot')
            self.colors = [self.cmap(i) for i in np.linspace(0, 1, 88)]
            self.sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=87))
            self.sm.set_array([])  # Needed for creating the colorbar

        self.input_wav_filename = input_wav_filename
        self.key_tones = [440.0 * (2 ** (i / 12.)) for i in range(-48, 40)]

    def load_data(self):
        '''Load wav data into np arrays'''
        sample_input = scipy.io.wavfile.read(self.input_wav_filename)
        sample_rate = sample_input[0]
        sample_data_channels = sample_input[1]
        sample_data = sample_data_channels[:, 0] # Only single-channel data supported
        sample_time = np.arange(len(sample_data)) / sample_rate
        self.sample_rate = sample_rate
        self.data_amps = sample_data
        self.data_times = sample_time
        return sample_data, sample_time, sample_rate

    def crop(self, sample_start_time=None, sample_end_time=None, display=None):
        '''Crop audio data'''
        data_amps = self.data_amps
        data_times = self.data_times
        if display is None:
            display = self.display

        if sample_start_time is None:
            sample_start_time = 0
        if sample_end_time is None:
            sample_end_time = data_times[-1]

        idx = np.where((data_times >= sample_start_time) & (data_times < sample_end_time))
        # sample_time, data_amps = sample_time[idx], data_amps[idx]
        # cropped_sample_time -= cropped_sample_time[0]
            
        if display:
            fig, axs = plt.subplots(2)
            ax = axs[0]
            ax.plot(data_times, data_amps, color="gray", label="input wav file data")
            ax.vlines([sample_start_time, sample_end_time], [np.min(data_amps)] * 2, [np.max(data_amps)] * 2, label="crop window")
            ax.set_xlabel('s')
            ax.legend()
            ax = axs[1]
            ax.plot(data_times[idx], data_amps[idx], 'r', label="cropped input data")
            ax.set_xlabel('s')
            plt.legend()
            plt.show()

        self.data_amps = data_amps[idx]
        self.data_times = data_times[idx]
        self.sample_start_time = sample_start_time
        self.sample_end_time = sample_end_time
        return self.data_amps, self.data_times

    def get_characteristic_key(self, display=None):
        if display is None:
            display = self.display
        print(f"Getting characteristic key signature of the piece...")
        matching_key = get_key_signature(self.input_wav_filename, display=display)
        print(f"Got charasteric key signature: {matching_key}")
        self.key_signature_frequency = matching_key
        return matching_key

    def get_characteristic_chord_duration(self, display=None):
        if display is None:
            display = self.display
        '''Take spectrum of the sample data and get strongest frequency between 1-10 Hz'''
        y, x = self.data_amps, self.data_times
        xf, yf = get_fft(x, y)
        idx = np.where((xf >= 0) & (xf < 20))
        xf, yf = xf[idx], yf[idx]
        # Process the data a bit to make the peak-finding easier
        conv_width = 0.05 # Convolve width in Hz
        yf_ = convolve_with_gaussian(xf, np.abs(yf)**2, conv_width)
        chord_freq_idx = np.argmax(yf_)
        chord_frequency = xf[chord_freq_idx]
        if display:
            fig, axs = plt.subplots(3, figsize=(5,8))
            ax = axs[0]
            ax.plot(x, y, 'b', label='audio data')
            ax.set_xlabel('s')
            ax.legend()
            ax = axs[1]
            ax.plot(xf, np.abs(yf), 'r', label='abs spectrum (low freqs)')
            ax.set_xlabel('Hz')
            ax.legend()
            ax = axs[2]
            ax.plot(xf, yf_, 'r', label=f'abs squared and conv width {conv_width}Hz')
            ax.plot(xf[chord_freq_idx], yf_[chord_freq_idx], 'g', marker='v', label='char. chord')
            ax.legend()
            ax.set_xlabel('Hz')
            plt.suptitle('Characteristic chord in low-frequency spectrum')
            plt.tight_layout()
            plt.show()
            print(f"Characteristic chord frequency is {round(chord_frequency, 3)} Hz (chord length {round(1/chord_frequency, 3)} s)")
        self.chord_frequency = chord_frequency
        self.chord_length = 1 / chord_frequency



    def get_chord_boundary_times(self, display=None, conv_width=0.05):
        """Find chord start times"""
        if display is None:
            display = self.display
        y = self.data_amps
        y = y / np.max(np.abs(y))
        x = self.data_times
        y_ = convolve_with_gaussian(x, np.abs(y)**2, conv_width)
        y_ = y_ / np.max(y_)
        dy = np.gradient(y_)
        dy = dy / np.max(np.abs(dy))
        if display:
            fig, axs = plt.subplots(2)
            ax = axs[0]
            ax.plot(x, y, 'r', label='sample')
            ax.legend()
            ax = axs[1]
            ax.plot(x, y_, 'r', alpha=0.5, label=f'processed width {conv_width}s)')
            ax.plot(x, dy, 'b', alpha=0.5, linewidth=0.5, label=f'gradient')
            ax.hlines(0, min(x), max(x), 'k', linewidth=0.5)
            ax.legend(fontsize=6)
            ax.set_xlabel('s')
            plt.tight_layout()
            plt.show()
    
    def play_sample(self, display=None, playback_speed=1.0, gif_delay=0.5):
        import sounddevice as sd
        import os
        import shutil
        import IPython

        if display is None:
            display = self.display
        
        # Define function to play audio
        def play_audio():
            sd.play(self.data_amps, self.sample_rate)
            sd.wait()

        def display_gif():
            IPython.display.display(IPython.display.HTML(f'<img src="sample.gif" />'))

        if display:
            x, y = self.data_times, self.data_amps
            tmp_folder = 'tmp/gif'
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            os.mkdir(tmp_folder)

            gif_time_step = 0.1
            gif_times = np.arange(self.sample_start_time, self.sample_end_time, gif_time_step)
            for gif_idx, gif_time in enumerate(gif_times):
                fig, ax = plt.subplots()
                ax.plot(x, y, 'b', label='Sample audio data')
                ax.plot([gif_time] * 2, [min(y), max(y)], 'k')
                ax.set_xlabel('s')
                ax.set_title(f't = {gif_time}s')
                gif_file_idx = "0" * (5 - len(str(gif_idx))) + str(gif_idx)
                plt.savefig(f"{tmp_folder}/{gif_file_idx}.png")
                plt.close()
            print(f"Saved gif figs")
            
            images_lst = [f'{tmp_folder}\\{x}' for x in os.listdir(tmp_folder)]
            tools.save_gif_from_images(
                images_lst,
                frame_duration_ms= gif_time_step / playback_speed,
                save_filename='sample.gif'
            )

        sd.play(self.data_amps, self.sample_rate * playback_speed)
        sd.wait()


    def simulate_harmonics(self, fundamental_freq, inharmonicity=0.02, num_harmonics=5):
        wave = np.zeros_like(self.data_times)
        # print("wave:",wave)
        for n in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * n * (1 + inharmonicity * np.random.randn())
            # print("harmonic_freq",harmonic_freq)
            wave += (1.0 / n) * np.sin(2.0 * np.pi * harmonic_freq * self.data_times)
        return wave


    def get_chord_profile(self, display=None, conv_width=None):
        if conv_width is None:
            conv_width = 0.05
        if display is None:
            display = self.display
        y = self.data_amps
        y = y / np.max(np.abs(y))
        x = self.data_times
        y_ = convolve_with_gaussian(x, np.abs(y)**2, conv_width)
        y_ = y_ / np.max(y_)

        print(f"Getting chord profile")
        duration = 3.0  # seconds
        sample_rate = self.sample_rate  # samples per second
        try:
            key_signature_freq = self.key_signature_frequency
        except:
            print(f"Characterizing key signature...")
            key_signature_freq = self.get_characteristic_key()

        # t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        t = np.arange(self.sample_start_time, self.sample_end_time, 1/self.sample_rate)
        key_signature_frequencies = tools.get_scale_tones(key_signature_freq['frequency'], octaves=[-1,0])

        alphas = [1, 3, 5]
        print("alphas",alphas)
        fig, axs = plt.subplots(3, len(alphas), sharex=True)
        axs = axs.flatten()
        for alpha_idx, alpha in enumerate(alphas):

            ax = axs[alpha_idx]
            ax.plot(x, y, 'r', label='sample data')
            ax.set_title(f'alpha {alpha}')

            total_wave = np.zeros_like(t)
            for delay in [0, 0.5, 0.8]:
                envelope = np.exp(-alpha * (t - t[0]))
                delay_idx = np.where(t - t[0] > delay)[0][0]
                print("delay_idx:",delay_idx)
                print("env:",envelope)
                for sig_freq in key_signature_frequencies:
                    wave = self.simulate_harmonics(sig_freq)
                    wave[delay_idx:] += wave[:len(wave)-delay_idx]
                    total_wave += wave * envelope
            total_wave = total_wave / np.max(np.abs(total_wave))

            ax = axs[alpha_idx + len(alphas)]
            ax.plot(t, total_wave, 'r', alpha=0.5, label='total wave')
            ax.legend()

            ax = axs[alpha_idx + 2 * len(alphas)]
            ax.plot(t, total_wave, 'r', alpha=0.5, label='simulated')
            ax.plot(x, y, 'b', alpha=0.5, label='data')
            ax.legend()

        plt.show()
        

    def track_notes(self, display=None, conv_width=None):
        """Focusing on key signature tones first, track their amplitudes across sample time.
        For the first iteration, don't distinguish different octaves.
        """
        if display is None:
            display = self.display
        if conv_width is None:
            conv_width = 0.05
        key_signature_tones = tools.get_scale_tones(self.key_signature_frequency)
        x, y = self.sample_times, self.sample_data
        xf, yf = get_fft(x, y)
        idx = np.where((xf >= 0) & (xf < 4000))
        xf, yf = xf[idx], yf[idx]
        # Process the data a bit to make the peak-finding easier
        yf_ = convolve_with_gaussian(xf, np.abs(yf)**2, conv_width)
        chord_freq_idx = np.argmax(yf_)
        if display:
            fig, ax = plt.subplots()
            ax.plot(x, y, 'r')



def add_note(time_array, amps_array, note_idx, note_time):
    """Get time series of amplitudes and add a note (note_idx [0, 87]) played at note_time"""

    return

def get_note_profile(time_data, amps_data):
    """Analyze time series of amplitudes and get the profile of the wave packet representing note(s) being played"""
    return

def simulate_note(time_array, note_idx, note_time):
    """Get the time series amplitudes of a note being played"""
    sample_rate = 3
    time_array = np.arange(0, 2, 1/sample_rate)
    return


if __name__=="__main__":
    args = parse_args(sys.argv[1:])
    wav_filename = args.wav_file
    conv_width = args.conv_width

    mt = musicTranscriber(wav_filename)

    y, x, hz = mt.load_data()
    mt.get_characteristic_key(display=False)
    mt.get_characteristic_chord_duration(display=False)

    mt.crop(sample_start_time=18, sample_end_time=19)
    mt.get_chord_profile(display=True, conv_width=0.5)

    # mt.get_chord_boundary_times(display=True, conv_width=0.005)
    # time.sleep(2)
    # mt.play_sample(display=False, playback_speed=.5)