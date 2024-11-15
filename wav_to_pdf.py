import scipy
import numpy as np
import matplotlib.pyplot as plt
import tools
import traceback
from scipy.signal import find_peaks
from key_signature_analysis import get_key_signature
import sys
from musicscore.beat import Beat
from musicscore.chord import Chord
from musicscore.measure import Measure
from musicscore.part import Part
from musicscore.score import Score
from musicscore.staff import Staff
from musicscore.voice import Voice
from musicscore.key import Key
from musicscore.note import Note
from musicscore.metronome import Metronome
import tools
import argparse
import os

import detect_notes

# # Import the wavfile data and metadata
# wav_filename = r"C:\Users\erica\OneDrive\Desktop\repos\music\Super Mario Bros. Theme  but its in a minor key.wav"
# sample_window_seconds = [0.838, 25]

# key_fifths_idx = get_key_signature(wav_filename, display=False)['fifths']

# sample_input = scipy.io.wavfile.read(wav_filename)
# sample_rate = sample_input[0]
# sample_data_channels = sample_input[1]
# sample_data = sample_data_channels[:, 0] # Only single-channel data supported
# idx_window = [int(sample_window_seconds[i] * sample_rate) for i in range(2)]
# sample_data = np.array(sample_data[idx_window[0]:idx_window[1]])
# sample_time = np.arange(len(sample_data)) / sample_rate
# sample_fft = tools.get_sorted_fft(sample_time, sample_data)
# xf, yf = sample_fft[0], sample_fft[1]
# sample_time_fft, sample_data_fft = tools.get_sorted_fft(sample_time, sample_data)

# plot = True

# dx = np.mean([sample_time[i+1] - sample_time[i] for i in range(len(sample_time)-2)])
# fontsz = 8

# # Take 3 random short samples
# sample_duration_seconds = 0.25
# sample_duration_num = int(sample_duration_seconds / dx)

# sample_window = [0, 2] # seconds
# sample_spacing_seconds = 0.1
# sample_freq_times = np.arange(sample_window[0], sample_window[1], sample_spacing_seconds) #np.linspace(0., 5., 20)
# sample_freqs = []
# freq_powers = []

# if plot:
#     fig, axs = plt.subplots(len(sample_freq_times), 3)
#     axs = axs.flatten()

# series_xf = []
# series_yf = []
# for i, gaussian_center in enumerate(sample_freq_times):
    
#     # To avoid distontinuous data, take a sample using gaussian distribution
#     gaussian_width = 0.5 * sample_duration_seconds
#     sampled_time_idx, sampled_data = tools.get_gaussian_sample(
#         sample_time, sample_data,
#         gaussian_center, gaussian_width,
#         crop_sigma = 3.0,
#     )
#     sampled_time = sample_time[sampled_time_idx]
    
#     xf = np.fft.fftfreq(len(sampled_data), d=dx)
#     yf = np.fft.fft(sampled_data)
#     series_xf.append(xf)
#     series_yf.append(yf)

#     peak_idxs, powers = tools.get_peak_frequencies(xf, yf)
#     peak_idxs = peak_idxs[:6]
#     peak_amps = yf[peak_idxs]
#     peak_freqs = xf[peak_idxs]
#     sample_freqs.append(peak_freqs)
#     freq_powers.append(powers)

#     # print(f"Random sample {i} peaks:")
#     # for peak_freq in peak_freqs:
#     #     name = tools.name_tone(peak_freq)
#     #     print(f"  frequency: {peak_freq}, name: {name['tone_name']}{name['octave']}, {round(name['cents'])} cents")
#     if plot:
#         print("i:",i)
#         plt_idx = i * 3
#         print("plt idx:",plt_idx)

#         ax = axs[plt_idx]
#         ax.plot(sample_time[sampled_time_idx], sample_data[sampled_time_idx], 'k', alpha=0.7)
#         ax.plot(sampled_time, sampled_data / max(sampled_data) * max(sample_data[sampled_time_idx]), 'b', alpha=0.4)
#         ax.set_title(f'{sample_duration_seconds}s gaussian sample at t={gaussian_center}s', fontsize=fontsz)
#         ax.tick_params(axis='both', labelsize=fontsz)

#         ax = axs[plt_idx + 1]
#         ax.plot(xf, yf.real, 'r')
#         ax.plot(xf, yf.imag, 'b')
#         ax.plot(xf, np.abs(yf), 'k')
#         ax.plot(xf[peak_idxs], np.abs(yf)[peak_idxs], 'gx')
#         ax.set_xlim([0, 2000])
#         ax.set_title('fft', fontsize=fontsz)
#         ax.tick_params(axis='both', labelsize=fontsz)
#         # ax.legend(fontsize=fontsz)

#         ax = axs[plt_idx + 2]
#         ax.plot(xf[peak_idxs], [10.* np.log(x) for x in powers], 'ro', linestyle='')


#         write_wavfilename = f'random_sample_{i}.wav'
#         scipy.io.wavfile.write(write_wavfilename, sample_rate, sampled_data.astype(np.int16))
#         print(f"Saved {write_wavfilename}")

# plt.show()

# print(sample_freqs)


def parse_args(args):

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'wav_file',
        type=str,
        # default="Super Mario Bros. Theme  but its in a minor key.wav",
        help='Name of .wav file (in current directory) to analyze key signature. If not specified, searches directory and uses first .wav file found.'
    )
    arg_parser.add_argument(
        '--frames-foldername',
        type=str,
        default='spectral_gif_images',
        help='Name of folder (in current directory) where frame images of final gif will be stored'
    )
    arg_parser.add_argument(
        '--sample-duration',
        type=float,
        default=0.07,
        help='Width of each sample in seconds'
    )
    arg_parser.add_argument(
        '-w',
        '--sample-window',
        type=float,
        nargs=2,
        default=[0, 8],
        help='Time range in seconds within which to take samples'
    )
    arg_parser.add_argument(
        '--sample-spacing',
        type=float,
        default=0.075,
        help='Take samples every this many seconds',
    )
    arg_parser.add_argument(
        '--xml-filename',
        type=str,
        default='mysheetmusic',
        help='Filename of .xml file to save'
    )
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    wav_filename = args.wav_file
    crop_wav_seconds = [0.838, 10]
    sample_duration = args.sample_duration
    sample_window = args.sample_window
    sample_spacing = args.sample_spacing
    save_frames_foldername = args.frames_foldername
    xml_filename = args.xml_filename
    create_gif_foldername = 'spectral_gif_images'

    data_time, data_amp = tools.preprocess_audio_data(
        wav_filename, crop_margin=crop_wav_seconds,
    )
    sample_times = np.arange(sample_window[0], sample_window[1], sample_spacing) #np.linspace(0., 5., 20)

    midi_detections, gif_plot_data = detect_notes.get_midi_detections(
        data_time, data_amp, sample_times, sample_duration,
        detection_threshold=2500000.0,
        create_gif_foldername=create_gif_foldername,
    )

    # Get the key signature of the audio
    key_signature = get_key_signature(wav_filename, display=False)
    key_fifths_idx = key_signature['fifths']
    key_frequency = key_signature['frequency']

    # Write it as sheet music xml
    beats_per_measure = 4
    chords_per_beat = 4

    s = Score(title=os.path.basename(wav_filename).replace('.wav',''))
    p = s.add_child(Part('piano', name='Piano'))
    key_signature = Key(fifths=key_fifths_idx)

    bpm = 60 / (sample_spacing * chords_per_beat)
    tempo = Metronome(bpm)
    middle_c = 440. * (2 ** (-9./12))

    
    chords_per_measure = beats_per_measure * chords_per_beat
    print("Chords per beat:",chords_per_measure)
    for idx in range(0, int(np.floor(len(midi_detections)/chords_per_measure) * chords_per_measure)):
        midis = midi_detections[idx]
        print("midis:", midis)
        if idx % chords_per_measure == 0:

            print(f"\nCreating new measure with 4 beats")
            m = p.add_child(Measure(number=None))
            m._key = key_signature

            ts = m.add_child(Staff(number=1))
            bs = m.add_child(Staff(number=2))
            tv = ts.add_child(Voice())
            bv = bs.add_child(Voice())

            tv.update_beats()
            bv.update_beats()
            print("Num beats: ",len(tv.get_children()))
        if idx != 0:
            m._time._show = False
        
        treble_chord = [x for x in midis if x >= 60]
        bass_chord = [x for x in midis if x < 60]
        print(f"  Treble, bass chords: {len(treble_chord)}, {len(bass_chord)}")
        
        beat_idx = int(
            np.floor((idx - 1) / 2) + (idx - 1) % 2
        ) % beats_per_measure
        print(f"Idx: {idx}, Beat idx: {beat_idx}")
        beat = tv.get_children()[beat_idx]
        if len(treble_chord) == 0:
            chord = Chord([0], 1. / chords_per_beat)
        else:
            chord = Chord(treble_chord, 1. / chords_per_beat)
            print(f"Added chord to treble beat")
        if idx == 0:
            chord._metronome = tempo
        beat.add_child(chord)
        
        beat = bv.get_children()[beat_idx]
        if len(bass_chord) == 0:
            chord = Chord([0], 1. / chords_per_beat)
        else:
            chord = Chord(bass_chord, 1. / chords_per_beat)
        if idx == 0:
            chord._metronome = tempo
        beat.add_child(chord)

    xml_path = f'{xml_filename}.xml'
    try:
        s.export_xml(xml_path)
        print(f"Wrote {xml_path}")

    except TypeError as e: # Expect a TypeError first time code is run
        # tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        print(f"{e}"
            "Encountered error in musescore package script when trying to write xml file (see error message above).\n"
            "This is expected the first time you run this code.\n"
            "Please comment out the 'if self.is_rest'... condition in chord.py \nat the location specified in the error message, save the file and try again.\n"
        )

