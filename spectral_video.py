import scipy
import numpy as np
import matplotlib.pyplot as plt
import tools
from scipy.signal import find_peaks
from key_signature_analysis import get_key_signature

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
import os

# Import the wavfile data and metadata
save_frames_foldername = 'spectral_gif_images'
curr_dir = os.getcwd()
save_frames_folderpath = os.path.join(curr_dir, save_frames_foldername)
if not os.path.exists(save_frames_folderpath):
    os.mkdir(save_frames_folderpath)
    print(f"Made new directory {save_frames_folderpath}")

wav_filename = r"C:\Users\erica\OneDrive\Desktop\repos\music\Super Mario Bros. Theme  but its in a minor key.wav"
sample_window_seconds = [0.838, 25]

key_signature = get_key_signature(wav_filename, display=False)
key_fifths_idx = key_signature['fifths']
key_frequency = key_signature['frequency']

sample_input = scipy.io.wavfile.read(wav_filename)
sample_rate = sample_input[0]
sample_data_channels = sample_input[1]
sample_data = sample_data_channels[:, 0] # Only single-channel data supported
idx_window = [int(sample_window_seconds[i] * sample_rate) for i in range(2)]
sample_data = np.array(sample_data[idx_window[0]:idx_window[1]])
sample_time = np.arange(len(sample_data)) / sample_rate
sample_fft = tools.get_fft(sample_time, sample_data)
xf, yf = sample_fft[0], sample_fft[1]
sample_time_fft, sample_data_fft = tools.get_fft(sample_time, sample_data)

dx = np.mean([sample_time[i+1] - sample_time[i] for i in range(len(sample_time)-2)])
fontsz = 8

# Take short samples
sample_duration_seconds = 0.25

sample_window = [0, 2] # seconds
sample_spacing_seconds = 0.05
sample_freq_times = np.arange(sample_window[0], sample_window[1], sample_spacing_seconds) #np.linspace(0., 5., 20)
sample_freqs = []
freq_powers = []

series_xf = []
series_yf = []
for i, gaussian_center in enumerate(sample_freq_times):
    
    # To avoid distontinuous data, take a sample using gaussian distribution
    gaussian_width = 0.5 * sample_duration_seconds
    sampled_time_idx, sampled_data = tools.get_gaussian_sample(
        sample_time, sample_data,
        gaussian_center, gaussian_width,
        crop_sigma = 3.0,
    )
    sampled_time = sample_time[sampled_time_idx]
    
    xf = np.fft.fftfreq(len(sampled_data), d=dx)
    yf = np.fft.fft(sampled_data)
    series_xf.append(xf)
    series_yf.append(np.abs(yf))

key_refs = tools.get_scale_tones(key_frequency, octaves=[-3,-2,-1,0,1,2,3])

max_amp = max([max(x) for x in series_yf])
for i, (xf, yf) in enumerate(zip(series_xf, series_yf)):
    fig, ax = plt.subplots()
    yf = yf / max(yf) 
    ax.plot(xf, yf)
    for key_ref in key_refs:
        ax.plot([key_ref, key_ref], [min(yf), max(yf)], 'r', alpha=0.5, linewidth=0.5)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1000])
    save_framepath = os.path.join(save_frames_folderpath, f'frame{i}.png')
    ax.set_title(f't={round(sample_freq_times[i],2)}')
    plt.savefig(save_framepath)
    print(f"Saved {save_framepath}")

images_lst = [f'{save_frames_folderpath}\\{x}' for x in os.listdir(save_frames_folderpath)]
tools.save_gif_from_images(images_lst, frame_duration_ms=2*sample_spacing_seconds*1000, save_filename='spectral_video.gif')



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
# bpm = 1 / sample_spacing_seconds * 60
# tempo = Metronome(bpm)
# middle_c = 440. * (2 ** (-9./12))


# for idx in range(0, int(np.ceil(len(sample_freqs)/4) * 4)):
#     sample_freq = sample_freqs[idx]
#     sample_freq_time = sample_freq_times[idx]
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
    # y_filtered_write = np.real(y_filtered / max(y_filtered) * max(sample_data))
    # ax.plot(x, tools.standardize(y_filtered), 'b', alpha=0.5)


    # ax.plot(xf, tools.standardize(yf_filtered), 'r', linestyle='dashed')
    # y_bass_write = np.real(y_filtered) / max(np.real(y_filtered)) * max(sample_data)
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
#     ax.plot(sample_time, sample_data)
#     ax.plot([offset]*2, [min(sample_data), max(sample_data)], 'r', linestyle='dashed', label='Offset')
#     ax.set_title('Sampled data', fontsize=fontsz)
#     ax.set_xlabel('t (s)', fontsize=fontsz)
#     ax.set_ylabel('amp', fontsize=fontsz)
#     ax.tick_params(axis='both', labelsize=fontsz)

#     for div_idx, div_seconds in enumerate(np.linspace(min_div_seconds, max_div_seconds, div_steps)):
#         trange = [offset, offset + div_seconds]
#         print(f"\nDIV SECONDS: {div_seconds}")
#         template_idx = np.where((sample_time >= trange[0]) & (sample_time < trange[-1]))[0]
#         template = tools.standardize(sample_data[template_idx])

#         ax = axs[div_idx+1]
#         ax.plot(template, color='b', alpha=0.4)
#         ax.set_title(f"{div_seconds}")

#         div_length = int(div_seconds / dx)
#         step_length = int(scan_step_seconds / dx)
#         idx_starts = range(template_idx[-1], len(sample_data) - div_length, step_length)
        
#         nxcorrs = []
#         for idx_start in idx_starts:
#             idx_end = idx_start + div_length
#             shifted_template = tools.standardize(sample_data[idx_start:idx_end])
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
#         best_template = tools.standardize(sample_data[idx_starts[best_xcorr_idx]:idx_starts[best_xcorr_idx] + div_length])
#         print(f"Best nxc: {nxcorrs[best_xcorr_idx]}")
#         ax.plot(best_template, alpha=0.4, color='r')
#         ax.set_title(f'Best xcorr: {nxcorrs[best_xcorr_idx]}', fontsize=fontsz)

# plt.show()


# sample_fft = tools.get_fft(sample_time, sample_data)
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

