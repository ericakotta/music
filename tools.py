import numpy as np
import scipy 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



HALFSTEP_RATIO = 2. ** (1. / 12)



def get_fft(x, y, comp=''):
    dx = np.mean([x[i+1] - x[i] for i in range(len(x)-1)])
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


def simulate_scale_spectrum(sampled_xf, tone=440, key='major', width=1, type='lorentzian'):
    '''Simulate spectrum of a sample with only tones of a certain key (major) are played'''
    simulated_yf = np.zeros(len(sampled_xf))
    tones = get_scale_tones(tone, key=key, octaves=[-3,3])
    for scale_tone in tones:
        if type.lower() == 'gaussian':
            single_tone_yf = np.exp( -1.0 * (scale_tone - sampled_xf) ** 2 / (2 * width**2))
        elif type.lower() == 'lorentzian':
            single_tone_yf = 1 / ( (sampled_xf - scale_tone) ** 2 + width ** 2)
        simulated_yf = simulated_yf + single_tone_yf
    simulated_yf = (simulated_yf - min(simulated_yf)) / (max(simulated_yf) - min(simulated_yf))
    return simulated_yf


def construct_scale(start_tone, key='major', offset=0):
    half_tone = start_tone / 12.0
    scale_steps = [0, 2, 4, 5, 7, 9, 11]
    scale_tones = [
        offset + start_tone + step * half_tone 
        for step in scale_steps
    ]
    return scale_tones


def get_scale_tones(start_tone, key='major', offset=0, octaves=[0]):

    if key.lower() == 'major':
        scale_idxs = [0, 2, 4, 5, 7, 9, 11]
    elif key.lower() == 'minor':
        scale_idxs = [0, 2, 3, 5, 7, 8, 10]
    
    scale_tones = [start_tone * (2 ** (i / 12.0)) for i in scale_idxs]
    scales_tones = []
    for octave in range(min(octaves), max(octaves)+1):
        scales_tones.extend([offset + x * (2 ** octave) for x in scale_tones])
    return np.array(scales_tones)


def write_tones_to_wav(tones_lst, tone_durations, sample_rate=44100, wav_filename='tones.wav', volume=1000, plot=False):
    if type(tone_durations) == list:
        tone_seconds = tone_durations
    else:
        tone_seconds = [tone_durations] * len(tones_lst)
    print("tones_lst: ",tones_lst)
    print("tone durations:", tone_seconds)
    y_tot = np.array([])
    for tone, seconds in zip(tones_lst, tone_seconds):
        seconds = round(seconds * tone) / tone
        x = np.arange(0, seconds, 1/sample_rate)
        y = np.sin(2 * np.pi * tone * x)
        y = y / np.max(y) * volume
        y_tot = np.concatenate((y_tot, y), axis=None)
        print("length y:",len(y))
    scipy.io.wavfile.write(wav_filename, sample_rate, y_tot.astype(np.int16))
    print(f"Saved waveform: {y_tot}")
    print(f"Saved file: {wav_filename}")

    if plot:
        fig, axs = plt.subplots()
        plt.plot(y_tot, 'ro')
        plt.show()



def plot_wav(wav_filename):
    wav_data = scipy.io.wavfile.read(wav_filename)
    sample_rate = wav_data[0]
    sample_data = wav_data[1]
    sample_time = np.arange(len(sample_data)) / sample_rate
    fig, axs = plt.subplots()
    plt.plot(sample_time, sample_data)
    plt.show()


def wrap_tone(tone):
    '''Wrap tone to middle (4) octave'''
    octave = 4
    octave_min = 440 * (2 ** (-9. / 12.)) # C4
    octave_max = 440 * (2 ** (3. / 12.)) # C5
    while tone < octave_min:
        tone *= 2.0
        octave -= 1
    while tone >= octave_max:
        tone /= 2.0
        octave += 1
    return tone, octave


def convolve_with_gaussian(x, y, sigma):
    dx = np.mean([x[i+1] - x[i] for i in range(len(x)-2)])
    gx = np.arange(-3*sigma, 3*sigma, dx)
    gaussian = np.exp(-(gx/sigma)**2/2)
    y_conv = np.convolve(y, gaussian, mode="same")
    y_conv = (y_conv - min(y_conv)) / (max(y_conv) - min(y_conv))
    return y_conv


def name_tone(tone):
    letter_names = [ 'C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B' ]
    letter_frequencies  = [440 * (2 ** (i / 12.0)) for i in range(-9, 3)]
    tone, octave = wrap_tone(tone)
    lower_tone = [x for x in letter_frequencies if x <= tone]
    if len(lower_tone) == 0:
        lower_tone = letter_frequencies[0]
    else:
        lower_tone = lower_tone[-1]
    tone_cents = 1200 * np.log2(tone / lower_tone)

    tone_idx = letter_frequencies.index(lower_tone)
    plus_or_minus = 'plus'
    if tone_cents >= 50:
        plus_or_minus = 'minus'
        tone_cents = tone_cents - 100
        tone_idx += 1
    if tone_idx == len(letter_names):
        tone_idx = 0
        octave += 1
    # print("tone idx:",tone_idx)
    # print("len letter names:",len(letter_names))

    letter_name = letter_names[tone_idx]
    tone_name = f"{letter_name}{octave} {round(tone_cents)} cents"
    # print(tone_name)
    tone_name_dict = {
        "tone_name": letter_name,
        "octave": octave,
        "offset_cents": tone_cents,
    }
    return tone_name_dict


def fold_in_spectrum(xf, yf, xf_window):
    ''' Collapse octave-notes to within xf_window '''
    idx = np.where((xf >= min(xf_window)) & (xf <= max(xf_window)))
    folded_xf = xf[idx]
    folded_yf = np.zeros(len(folded_xf))
    return folded_yf


def normalize(y):
    return (y - min(y)) / (max(y) - min(y))

def standardize(y):       
    return (y - np.mean(y)) / np.std(y)



def get_peak_frequencies(xf, yf):
    yf = standardize(yf)
    
    return

def wrap_spectrum(xf, yf, xrange=[]):
    xrange = [0, 1000]
    if not xrange:
        xrange = [440 * (2 ** (-9. / 12)), 400 * (2 ** (3. / 12))]
        print(f"xrange: {xrange}")

    dx = np.mean([xf[i+1] - xf[i] for i in range(len(xf)-2)])

    idx = np.where((xf >= xrange[0]) & (xf < xrange[1]))
    print("idx:",idx)
    xf_, yf_ = xf[idx], yf[idx]
    # xf_ = np.log2(xf) + xf / 12.
    yf_ = (yf_ - min(yf_)) / (max(yf_) - min(yf_))

    peaks, props = find_peaks(yf_, prominence=0.5)
    proms = props['prominences']
    # sort by prominence
    sort_idx = np.flip(np.argsort(proms))
    peaks, frequencies, peak_proms = peaks[sort_idx], xf_[peaks[sort_idx]], proms[sort_idx]

    peak_frequencies = np.array([frequencies[0]])
    peak_idxs = np.array([peaks[0]])

    half_down, half_up = [2 ** (i / 12.) for i in [-1, 1]]
    for peak, frequency in zip(peaks[1:], frequencies[1:]):
        peak_ratios = peak_frequencies / frequency
        close_peaks = np.where(
            (peak_ratios > half_down) & (peak_ratios < half_up)
        )[0]
        print("close peaks:",close_peaks)
        if len(close_peaks) == 0:
            print("asdf")
            peak_frequencies = np.append(peak_frequencies, frequency)
            peak_idxs = np.append(peak_idxs, peak)

    peak_frequencies
    peak_idxs

    fig, ax = plt.subplots(2)
    ax[0].plot(xf_, yf_)
    ax[0].plot(xf_[peak_idxs], yf_[peak_idxs], 'rx')
    ax[0].set_xlim([0, 1000])


    # for idx, peak in enumerate(peaks):
    #     print(f"idx: {peak}, freq: {xf_[peak]}")
    #     print(props['prominences'][idx])

    # ax[1].plot(xf_, yf)
    plt.show()
    wrapped_yf = yf[idx]
    wrapped_xf = xf[idx]


    
