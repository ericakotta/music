import scipy
import numpy as np
import matplotlib.pyplot as plt
import tools
from scipy.signal import find_peaks
from key_signature_analysis import get_key_signature
import argparse
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
import sys
import os



def parse_args(args):

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--wav-file',
        type=str,
        default="Super Mario Bros. Theme  but its in a minor key.wav",
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
        default=0.1,
        help='Width of each sample in seconds'
    )
    arg_parser.add_argument(
        '--sample-range',
        type=float,
        nargs=2,
        default=[0, 3],
        help='Time range in seconds within which to take samples'
    )
    arg_parser.add_argument(
        '--sample-spacing',
        type=float,
        default=0.1,
        help='Take samples every this many seconds',
    )
    arg_parser.add_argument(
        '--make-gif',
        action='store_true',
        help='Create gif of results.'
    )
    return arg_parser.parse_args()



def get_midi_detections(
        data_time, data_amp, sample_times, sample_duration,
        detection_threshold=None, create_gif_foldername=False,
    ):
    # Define the 88 piano key frequencies
    key_tones = [440.0 * (2 ** (i / 12.)) for i in range(-48, 40)]

    frequencies = []
    spectral_amps = []
    spectral_amp_changes = []
    keys_spectral_changes = []
    for i, gaussian_center in enumerate(sample_times):
        '''Spectrum at each sample time'''
        # To avoid distontinuous data, take a sample using gaussian distribution
        gaussian_width = 0.5 * sample_duration
        _, sampled_data = tools.get_gaussian_sample(
            data_time, data_amp,
            gaussian_center, gaussian_width,
        )
        dx = np.mean([data_time[i+1] - data_time[i] for i in range(len(data_time)-1)])
        frequency = np.fft.fftfreq(len(sampled_data), d=dx)
        spectral_amp = np.fft.fft(sampled_data)
        # Sort with respect to frequency
        idx = np.argsort(frequency)
        frequency, spectral_amp = frequency[idx], spectral_amp[idx]
        frequencies.append(frequency)
        spectral_amps.append(np.abs(spectral_amp))

        '''Then take the change in spectrum from previous sample time'''
        if i == 0:
            spectral_amp_change = np.zeros(spectral_amp.shape)
        elif i > 0:
            spectral_amp_change = spectral_amps[-1] - spectral_amps[-2]
        spectral_amp_changes.append(spectral_amp_change)

        spectral_change_per_key = np.zeros(len(key_tones))
        for key_idx, key_tone in enumerate(key_tones):
            min_freq = key_tone * (2 ** (-1.0 / 24))
            max_freq = key_tone * (2 ** (1.0 / 24))
            spectral_change_window = spectral_amp_change[
                np.where((frequency >= min_freq) & (frequency < max_freq))
            ]
            spectral_change_per_key[key_idx] = max(spectral_change_window)
        keys_spectral_changes.append(spectral_change_per_key)
        
    if detection_threshold is None:
        detection_threshold = 1500000

    # Plot histogram of per-key spectral amp changes
    keys_spectral_changes_values = []
    for i, gaussian_center in enumerate(sample_times):
        '''Analysis to determine a good note-detection threshold'''
        for spectral_change_per_key in keys_spectral_changes:
            keys_spectral_changes_values.extend(spectral_change_per_key)
    plt.hist(keys_spectral_changes_values, bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Spectral amp change')
    plt.ylabel('Count')
    plt.title('Per key spectral changes')
    plt.show()

    '''Analyze spectral change to determine a good threshold'''
    midi_detections = []
    midi_antidetections = []
    for keys_spectral_change in keys_spectral_changes:
        detected_notes = np.array(key_tones)[
            np.where(keys_spectral_change >= detection_threshold)
        ]
        antidetected_notes = np.array(key_tones)[
            np.where(keys_spectral_change <= - detection_threshold)
        ]
        midi_detections.append(tools.frequency_to_midi(detected_notes))
        midi_antidetections.append(tools.frequency_to_midi(antidetected_notes))
    
    gif_plot_data = {
        'frequencies': frequencies,
        'spectral_amps': spectral_amps,
        'spectral_amp_changes': spectral_amp_changes,
        'midi_detections': midi_detections,
        'detection_times': sample_times,
        'midi_antidetections': midi_antidetections,
    }

    if create_gif_foldername:
        frame_duration_s = 2 * (sample_times[-1] - sample_times[0]) / (len(sample_times) - 1)

        save_plot_pngs(data_time, data_amp, gif_plot_data, sample_times, create_gif_foldername)
        images_lst = [f'{create_gif_foldername}\\{x}' for x in os.listdir(create_gif_foldername)]
        tools.save_gif_from_images(
            images_lst,
            frame_duration_ms=frame_duration_s*1000,
            save_filename='spectral_detections.gif'
        )
    return midi_detections, gif_plot_data


def save_plot_pngs(data_time, data_amp, gif_data, gif_data_times, save_frames_folderpath):
    '''Save series of snapshots of audio data, spectra, and spectral change across sampled times'''
    # Define 88 piano keys for plotting reference
    key_tones = [440.0 * (2 ** (i / 12.)) for i in range(-48, 40)]
    fontsz = 8

    # Import the data
    frequencies = gif_data['frequencies']
    amps = gif_data['spectral_amps']
    amp_diffs = gif_data['spectral_amp_changes']
    midi_detections_list = gif_data['midi_detections']
    midi_antidetections_list = gif_data['midi_antidetections']
    print("midi detections",midi_detections_list)
    # Prepare directories
    exist_files = [f"{save_frames_folderpath}\\{x}" for x in os.listdir(save_frames_folderpath)]
    if len(exist_files) > 0:
        print(f"Deleting {len(exist_files)} existing images in {save_frames_folderpath}.")
        for exist_file in exist_files:
            os.remove(exist_file)
    
    print(f"Saving gif frame images to {save_frames_folderpath}...")
    max_amp = max([max(x) for x in amps])
    max_amp_diff = max([max(np.abs(x)) for x in amp_diffs])
    plot_data_idx = np.where((data_time >= gif_data_times[0]) & (data_time <= gif_data_times[-1]))
    

    for i, (frequency, amp, amp_diff, midi_detections, midi_antidetections) in enumerate(
        zip(frequencies, amps, amp_diffs, midi_detections_list, midi_antidetections_list)
    ):
        midi_detection_freqs = tools.midi_to_frequency(midi_detections)
        midi_detection_amps = np.interp(midi_detection_freqs, frequency, amp_diff)
        midi_antidetection_freqs = tools.midi_to_frequency(midi_antidetections)
        midi_antidetection_amps = np.interp(midi_antidetection_freqs, frequency, amp_diff)


        fig, axs = plt.subplots(3)

        ax = axs[0]
        ax.plot(data_time[plot_data_idx], data_amp[plot_data_idx], 'k')
        ax.plot([gif_data_times[i]] * 2, [min(data_amp[plot_data_idx]), max(data_amp[plot_data_idx])], 'r')
        ax.set_xlabel('t (s)', fontsize=fontsz)
        ax.set_ylabel('amp', fontsize=fontsz)
        ax.tick_params(axis='both', labelsize=fontsz)

        ax = axs[1]
        ax.plot(frequency, amp, 'b')
        for x in key_tones:
            ax.plot([x] * 2, [0, max_amp], 'r', alpha=0.5, linewidth=0.5)
        ax.set_ylim([0, max_amp])
        ax.set_xlim([0, 1000])
        ax.set_title(f'spectrum t={round(gif_data_times[i],2)}', fontsize=fontsz)
        ax.set_xlabel('t (s)', fontsize=fontsz)
        ax.set_ylabel('fft amp', fontsize=fontsz)
        ax.tick_params(axis='both', labelsize=fontsz)

        ax = axs[2]
        ax.plot(frequency, amp_diff, 'b')
        ax.plot(midi_detection_freqs, midi_detection_amps, 'r.')
        ax.plot(midi_antidetection_freqs, midi_antidetection_amps, 'g.')
        for x in key_tones:
            ax.plot([x] * 2, [-max_amp_diff, max_amp_diff], 'r', alpha=0.5, linewidth=0.5)
        ax.set_ylim([-max_amp_diff, max_amp_diff])
        ax.set_xlim([0, 1000])
        ax.set_title(f'spectrum change t={round(gif_data_times[i], 2)}', fontsize=fontsz)
        ax.set_xlabel('t (s)', fontsize=fontsz)
        ax.set_ylabel('delta fft amp', fontsize=fontsz)
        ax.tick_params(axis='both', labelsize=fontsz)

        frame_idx_str = '0' * (5 - len(str(i))) + str(i)
        save_framepath = os.path.join(save_frames_folderpath, f'frame_{frame_idx_str}.png')
        plt.tight_layout()
        plt.savefig(save_framepath)
        plt.close()

    print(f"Saved {len(amp_diffs)} images to {save_frames_folderpath}.")


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    wav_filename = args.wav_file
    crop_wav_seconds = [0.838, 10]
    sample_duration = args.sample_duration
    sample_window = args.sample_range
    sample_spacing = args.sample_spacing
    save_frames_foldername = args.frames_foldername
    make_gif = args.make_gif

    # Import the wavfile data and metadata
    data_time, data_amp = tools.preprocess_audio_data(
        wav_filename, crop_margin=crop_wav_seconds,
    )
    # Define the 88 piano tones
    key_tones = [440.0 * (2 ** (i / 12.)) for i in range(-48, 40)]

    # Set up directory for saving gif
    curr_dir = os.getcwd()
    save_frames_folderpath = os.path.join(curr_dir, save_frames_foldername)
    if not os.path.exists(save_frames_folderpath):
        os.mkdir(save_frames_folderpath)
        print(f"Made new directory {save_frames_folderpath}")

    sample_times = np.arange(sample_window[0], sample_window[1], sample_spacing) #np.linspace(0., 5., 20)

    midi_detections, gif_plot_data = get_midi_detections(data_time, data_amp, sample_times, sample_duration)
    
    if make_gif:
        save_plot_pngs(data_time, data_amp, gif_plot_data, sample_times, save_frames_folderpath)

        images_lst = [f'{save_frames_folderpath}\\{x}' for x in os.listdir(save_frames_folderpath)]
        tools.save_gif_from_images(
            images_lst,
            frame_duration_ms=2*sample_spacing*1000,
            save_filename='spectral_video.gif'
        )

    # sample_times = np.arange(sample_window[0], sample_window[1], sample_spacing) #np.linspace(0., 5., 20)
    # sample_freqs = []
    # freq_powers = []
    # frequencies = []
    # spectral_amps = []
    # midi_detections = []
    # spectral_amp_changes = []
    # for i, gaussian_center in enumerate(sample_times):
    #     '''Spectrum at each sample time'''
    #     # To avoid distontinuous data, take a sample using gaussian distribution
    #     gaussian_width = 0.5 * sample_duration
    #     sampled_time_idx, sampled_data = tools.get_gaussian_sample(
    #         data_time, data_amp,
    #         gaussian_center, gaussian_width,
    #         # crop_sigma = 3.0,
    #     )
    #     sampled_time = data_time[sampled_time_idx]
    #     dx = np.mean([data_time[i+1] - data_time[i] for i in range(len(data_time)-1)])
    #     frequency = np.fft.fftfreq(len(sampled_data), d=dx)
    #     spectral_amp = np.fft.fft(sampled_data)
    #     idx = np.argsort(frequency)
    #     frequency, spectral_amp = frequency[idx], spectral_amp[idx]
    #     frequencies.append(frequency)
    #     spectral_amps.append(np.abs(spectral_amp))

    #     '''Then take the change in spectrum from previous sample time'''
    #     if i == 0:
    #         spectral_amp_change = np.zeros(spectral_amp.shape)
    #     elif i > 0:
    #         spectral_amp_change = spectral_amps[-1] - spectral_amps[-2]
    #     spectral_amp_changes.append(spectral_amp_change)

    #     spectral_change_per_key = np.zeros(len(key_tones))
    #     threshold = 2500000.0
    #     # gaussian_x = np.arange(-3. * sigma, 3 * sigma, x_step)
    #     for key_idx, key_tone in enumerate(key_tones):
    #         min_freq = key_tone * (2 ** (-1.0 / 24))
    #         max_freq = key_tone * (2 ** (1.0 / 24))
    #         spectral_change_window = spectral_amp_change[
    #             np.where((frequency >= min_freq) & (frequency < max_freq))
    #         ]
    #         spectral_change_per_key[key_idx] = max(spectral_change_window)
        
    #     detected_notes = np.array(key_tones)[np.where(spectral_change_per_key >= threshold)]
    #     midi_detections.append(tools.frequency_to_midi(detected_notes))
        



# key_detections = np.zeros(len(fft_y_diffs))


# for i, (frequency, spectral_change) in enumerate(zip(fft_x_series, fft_y_diffs)):
#     idx = np.argsort(fft_x)
#     key_diff = np.interp(key_tones, spectral_change[idx], spectral_change[idx])
#     if i == 3:
        
        
#         # Find the max spectral change at each frequency
#         # Take sum of gaussian of width sigma centered at frequency times spectral change
#         # sigma = 1
#         x_step = np.mean([np.abs(frequency[i+1] - frequency[i]) for i in range(len(frequency)-  2)])
#         key_spectral_change = []
#         # gaussian_x = np.arange(-3. * sigma, 3 * sigma, x_step)
#         for key_tone in key_tones:
#             min_freq = key_tone * (2 ** (-1.0 / 24))
#             max_freq = key_tone * (2 ** (1.0 / 24))
#             spectral_change_window = spectral_change[
#                 np.where((frequency >= min_freq) & (frequency < max_freq))
#             ]
#             key_spectral_change.append(max(spectral_change_window))

#         plt.plot(frequency, spectral_change, 'k')
#         plt.plot(key_tones, key_spectral_change, 'r.')
#         for key_tone in key_tones:
#             plt.plot([key_tone] * 2, [min(key_spectral_change), max(key_spectral_change)], 'r', alpha=0.5, linewidth=0.5)
#         plt.xlim(min(key_tones) * (2 ** (-1.0 / 24)), max(key_tones) * (2 ** (1.0 / 24)))
# plt.show()


'''

'''

# '''Calculate the spectral change across time'''
# fft_y_changes = []
# convolve_sigma_hz = 0.2
# for i in range(1, len(fft_y_series) - 1):
#     fft_x = fft_x_series[i]
#     fft_y = fft_y_series[i]
#     fft_y0 = fft_y_series[i - 1]

#     # Convolve to dampen spikes from noise
#     fft_y = tools.convolve_with_gaussian(fft_x, fft_y, convolve_sigma_hz)
#     fft_y0 = tools.convolve_with_gaussian(fft_x, fft_y0, convolve_sigma_hz)
#     fft_y_change = fft_y - fft_y0
#     fft_y_changes.append(fft_y_change)





#     print("i:",i)
#     plt_idx = i * 2
#     print("plt idx:",plt_idx)
#     fig, ax = plt.subplots()

#     ax.plot(xf, np.abs(yf), 'k')
#     ax.set_xlim([0, 1000])
#     ax.set_title('fft', fontsize=fontsz)
#     ax.tick_params(axis='both', labelsize=fontsz)

#     # ax = axs[plt_idx + 2]
#     # ax.plot(xf[peak_idxs], [10.* np.log(x) for x in powers], 'ro', linestyle='')


#     write_wavfilename = f'random_sample_{i}.wav'
#     scipy.io.wavfile.write(write_wavfilename, sample_rate, sampled_data.astype(np.int16))
#     print(f"Saved {write_wavfilename}")

# plt.show()

# print(sample_freqs)

# # Write it as sheet music xml
# s = Score(title=f"random_sample")
# p = s.add_child(Part('piano', name='Piano'))
# key_signature = Key(fifths=key_fifths_idx)
# bpm = 1 / sample_spacing * 60
# tempo = Metronome(bpm)
# middle_c = 440. * (2 ** (-9./12))


# for idx in range(0, int(np.ceil(len(sample_freqs)/4) * 4)):
#     sample_freq = sample_freqs[idx]
#     sample_freq_time = sample_times[idx]
#     if idx % 4 == 0:

#         print(f"\nCreating new measure with 4 beats")
#         m = p.add_child(Measure(number=None))
#         m._key = key_signature

#         ts = m.add_child(Staff(number=1))
#         bs = m.add_child(Staff(number=2))
#         tv = ts.add_child(Voice())
#         bv = bs.add_child(Voice())

#         tv.update_beats()
#         bv.update_beats()


#     if idx != 0:
#         m._time._show = False
    
#     treble_chord = [x for x in sample_freq if x > middle_c]
#     bass_chord = [x for x in sample_freq if x <= middle_c]
#     print(f"  Treble, bass chords: {len(treble_chord)}, {len(bass_chord)}")
    
#     beat = tv.get_children()[idx % 4]
#     if len(treble_chord) == 0:
#         chord = Chord([0], 1)
#     else:
#         chord = Chord(tools.frequency_to_midi(treble_chord),1)
#         print(f"Added chord to treble beat")
#     if idx == 0:
#         chord._metronome = tempo
#     beat.add_child(chord)
    
#     beat = bv.get_children()[idx % 4]
#     if len(bass_chord) == 0:
#         chord = Chord([0], 1)
#     else:
#         chord = Chord(tools.frequency_to_midi(bass_chord), 1)
#     if idx == 0:
#         chord._metronome = tempo
#     beat.add_child(chord)


# xml_path = 'dummy_test1.xml'
# s.export_xml(xml_path)
# print(f"Wrote {xml_path}")


    # Filter out some frequencies
    # yf_filtered = yf.copy()
    # yf_filtered[np.where(np.abs(xf) <= 250)] = 0
    # y_filtered = np.fft.ifft(yf_filtered)
    # y_filtered = tools.standardize(y_filtered)
    # y_filtered_write = np.real(y_filtered / max(y_filtered) * max(data_amp))
    # ax.plot(x, tools.standardize(y_filtered), 'b', alpha=0.5)


    # ax.plot(xf, tools.standardize(yf_filtered), 'r', linestyle='dashed')
    # y_bass_write = np.real(y_filtered) / max(np.real(y_filtered)) * max(data_amp)
    # scipy.io.wavfile.write(f'random_sample_{i}_bass.wav', sample_rate, y_filtered_write.astype(np.int16))


# plt.plot(y, 'k', alpha=0.5)
# plt.plot(y_bass_write, 'r', alpha=0.5)



# # Search for a section that repeats
# min_div_seconds, max_div_seconds = [5, 7]
# div_steps = 3
# div_step_seconds = 0.1
# scan_step_seconds = 1.0

# offset_seconds = np.linspace(2.4, 2.8, 5)
# for offset in offset_seconds:
#     print(f"OFFSET: {offset}s")

#     fig, axs = plt.subplots(5,2)
#     axs = axs.flatten()

#     ax = axs[0]
#     ax.plot(data_time, data_amp)
#     ax.plot([offset]*2, [min(data_amp), max(data_amp)], 'r', linestyle='dashed', label='Offset')
#     ax.set_title('Sampled data', fontsize=fontsz)
#     ax.set_xlabel('t (s)', fontsize=fontsz)
#     ax.set_ylabel('amp', fontsize=fontsz)
#     ax.tick_params(axis='both', labelsize=fontsz)

#     for div_idx, div_seconds in enumerate(np.linspace(min_div_seconds, max_div_seconds, div_steps)):
#         trange = [offset, offset + div_seconds]
#         print(f"\nDIV SECONDS: {div_seconds}")
#         template_idx = np.where((data_time >= trange[0]) & (data_time < trange[-1]))[0]
#         template = tools.standardize(data_amp[template_idx])

#         ax = axs[div_idx+1]
#         ax.plot(template, color='b', alpha=0.4)
#         ax.set_title(f"{div_seconds}")

#         div_length = int(div_seconds / dx)
#         step_length = int(scan_step_seconds / dx)
#         idx_starts = range(template_idx[-1], len(data_amp) - div_length, step_length)
        
#         nxcorrs = []
#         for idx_start in idx_starts:
#             idx_end = idx_start + div_length
#             shifted_template = tools.standardize(data_amp[idx_start:idx_end])
#             xcorr = np.sum(np.multiply(template, shifted_template))
#             nxcorr = (1 / len(template)) * (
#                 np.sum(np.multiply(template, shifted_template))
#             ) / (
#                 np.sqrt(np.sum(template ** 2)) * np.sqrt(np.sum(shifted_template ** 2))
#             )
#             nxcorrs.append(xcorr)
#             if len(template) != len(shifted_template):
#                 print(f"Len template: {len(template)}, shifted: {len(shifted_template)}")

#         best_xcorr_idx = np.argmax(nxcorrs)
#         best_template = tools.standardize(data_amp[idx_starts[best_xcorr_idx]:idx_starts[best_xcorr_idx] + div_length])
#         print(f"Best nxc: {nxcorrs[best_xcorr_idx]}")
#         ax.plot(best_template, alpha=0.4, color='r')
#         ax.set_title(f'Best xcorr: {nxcorrs[best_xcorr_idx]}', fontsize=fontsz)

# plt.show()


# sample_fft = tools.get_fft(data_time, data_amp)
# xf, yf = sample_fft[0], sample_fft[1]
# ax = axs[1]
# ax.plot(xf, yf.real, 'b', label='real')
# ax.plot(xf, yf.imag, 'r', label='imag')
# ax.plot(xf, np.abs(yf), 'k', label='abs')
# ax.plot(xf, np.abs(yf), 'k', label='abs')
# ax.legend(fontsize=fontsz)
# ax.set_title('Sampled data spectrum', fontsize=fontsz)
# ax.set_xlabel('Hz', fontsize=fontsz)
# ax.set_ylabel('amp', fontsize=fontsz)
# ax.tick_params(axis='both', labelsize=fontsz)
# ax.set_xlim([-3000, 3000])

# see_idx = np.where((xf >= 256) & (xf <= 512))#xf.index(see_x[0]), xf.index(see_x[1])
# plt_xf = np.array(xf)[see_idx]
# plt_yf = np.array(np.abs(yf))[see_idx]

# key_steps = 20
# key_tones = [440 * (2 ** (i / 12.)) for i in range(-9, 3)]
# middle_tones = [440 * (2 ** (i / 12.)) for i in range(-9, 3)]
# key_tones = []
# for tone in middle_tones:
#     key_tones.extend([tone * (2 ** (i / (12 * key_steps))) for i in range(key_steps)])

# meas_yf = (plt_yf - min(plt_yf)) / (max(plt_yf) - min(plt_yf))
# conv_sigma = 0.2
# meas_yf_conv = tools.convolve_with_gaussian(plt_xf, meas_yf, sigma=conv_sigma)
# scores = []
# sim_width = 2
# for key_tone in key_tones:
#     sim_yf = tools.simulate_scale_spectrum(plt_xf, tone=key_tone, width=sim_width)
#     score = np.sum(np.multiply(meas_yf_conv, sim_yf)**2)
#     scores.append(score)

# best_match_idx = scores.index(max(scores))
# best_match_tone = key_tones[best_match_idx]
# best_match_yf = tools.simulate_scale_spectrum(plt_xf, tone=best_match_tone, width=sim_width)

# ax = axs[2]
# ax.plot(plt_xf, best_match_yf, 'r', alpha=1, label=f"sim {round(best_match_tone,1)}Hz")
# ax.plot(plt_xf, meas_yf, 'k', label=f"meas")
# ax.plot(plt_xf, meas_yf_conv, 'g', label=f"meas conv {conv_sigma}Hz")
# ax.set_title(f'Spectrum zoom', fontsize=fontsz)
# ax.legend(fontsize=fontsz)
# ax.set_xlabel('Hz', fontsize=fontsz)
# ax.set_ylabel('amp', fontsize=fontsz)
# ax.tick_params(axis='both', labelsize=fontsz)

# key_major = best_match_tone
# key_minor = key_major * (2 ** (-3. / 12.))
# key_major_name = tools.name_tone(key_major)
# key_minor_name = tools.name_tone(key_minor)
# # print(f"{round(key_major,1)} Hz, {key_major_name} (major) / {key_minor_name} (minor)")
# # match_key_label = f"{key_major_name.split(' ')[0].upper()} (Major) / {key_minor_name.split(' ')[0].lower()} (minor)"
# # axs[3].plot(key_tones, scores, 'ro')

# ax = axs[3]
# ax.plot(key_tones, scores, 'ro', linestyle='dotted')
# ax.plot(key_tones[best_match_idx], scores[best_match_idx], 'bx', label=key_major_name["tone_name"])
# ax.legend(fontsize=fontsz)
# ax.set_xlabel('Sim spectrum scale Hz', fontsize=fontsz)
# ax.set_ylabel('Similarity', fontsize=fontsz)
# ax.set_title('Similarity btw meas spectrum and sim spectrum', fontsize=fontsz)
# ax.tick_params(axis='both', labelsize=fontsz)

# plt.suptitle(f'Matching key: {key_major_name["tone_name"]} Major / {key_minor_name["tone_name"]} minor', fontsize=1.5*fontsz, fontweight="bold")
# plt.tight_layout()

# plt.show()

