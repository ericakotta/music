import scipy
import numpy as np
import matplotlib.pyplot as plt
import tools

# Import the wavfile data and metadata
wav_filename = r"C:\Users\erica\OneDrive\Desktop\repos\music\Super Mario Bros. Theme  but its in a minor key.wav"
sample_input = scipy.io.wavfile.read(wav_filename)
sample_rate = sample_input[0]
sample_data_channels = sample_input[1]
sample_data = sample_data_channels[:, 0] # Only single-channel data supported
sample_time = np.arange(len(sample_data)) / sample_rate
sample_fft = tools.get_fft(sample_time, sample_data)
xf, yf = sample_fft[0], sample_fft[1]
sample_time_fft, sample_data_fft = tools.get_fft(sample_time, sample_data)

fontsz = 8
fig, axs = plt.subplots(2,2)
axs = axs.flatten()

ax = axs[0]
ax.plot(sample_time, sample_data)
ax.set_title('Sampled data', fontsize=fontsz)
ax.set_xlabel('t (s)', fontsize=fontsz)
ax.set_ylabel('amp', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)

ax = axs[1]
ax.plot(sample_time_fft, sample_data_fft.real, 'b', label='real')
ax.plot(sample_time_fft, sample_data_fft.imag, 'r', label='imag')
ax.plot(sample_time_fft, np.abs(sample_data_fft), 'k', label='abs')
ax.legend(loc='upper right', fontsize=fontsz)
ax.set_title('Sampled data fft ', fontsize=fontsz)
ax.set_xlabel('frequency (Hz)', fontsize=fontsz)
ax.set_ylabel('amp', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)
ax.set_xlim([-3000, 3000])

# Zoom in to middle octave
zoom_frequencies = [440.0 * (2 ** (i / 12.)) for i in [-9.0, 3.0]]
zoom_idx = np.where((sample_time_fft >= zoom_frequencies[0]) & (xf < zoom_frequencies[1]))
plt_xf = np.array(sample_time_fft)[zoom_idx]
plt_yf = np.array(np.abs(yf))[zoom_idx]

# Define keys to simulate spectrum for and compare to measured spectrum
key_steps = 20
key_tones = [440 * (2 ** (i / 12.)) for i in range(-9, 3)]
middle_tones = [440 * (2 ** (i / 12.)) for i in range(-9, 3)]
key_tones = []
for tone in middle_tones:
    key_tones.extend([tone * (2 ** (i / (12 * key_steps))) for i in range(key_steps)])

# Convolute the measured spectrum for better alignment
meas_yf = (plt_yf - min(plt_yf)) / (max(plt_yf) - min(plt_yf))
conv_sigma = 0.2
meas_yf_conv = tools.convolve_with_gaussian(plt_xf, meas_yf, sigma=conv_sigma)
scores = []
# TODO: Characterize width of measured spectral peak widths and use that for sim_width
sim_width = 2
for key_tone in key_tones:
    sim_yf = tools.simulate_scale_spectrum(plt_xf, tone=key_tone, width=sim_width)
    score = np.sum(np.multiply(meas_yf_conv, sim_yf)**2)
    scores.append(score)
# Find the tone of best matching simulated spectrum
best_match_idx = scores.index(max(scores))
best_match_tone = key_tones[best_match_idx]
best_match_yf = tools.simulate_scale_spectrum(plt_xf, tone=best_match_tone, width=sim_width)
key_major_name = tools.name_tone(best_match_tone)
key_minor_name = tools.name_tone(best_match_tone * (2 ** (-3. / 12.)))

# Plot results
ax = axs[2]
ax.plot(plt_xf, best_match_yf, 'r', alpha=1, label=f"sim {round(best_match_tone,1)}Hz")
ax.plot(plt_xf, meas_yf, 'k', label=f"meas")
ax.plot(plt_xf, meas_yf_conv, 'g', label=f"meas conv {conv_sigma}Hz")
ax.set_title(f'Spectrum zoom', fontsize=fontsz)
ax.legend(fontsize=fontsz)
ax.set_xlabel('Hz', fontsize=fontsz)
ax.set_ylabel('amp', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)

ax = axs[3]
ax.plot(key_tones, scores, 'ro', linestyle='dotted')
ax.plot(key_tones[best_match_idx], scores[best_match_idx], 'bx', label=key_major_name["tone_name"])
ax.legend(fontsize=fontsz)
ax.set_xlabel('Sim spectrum scale Hz', fontsize=fontsz)
ax.set_ylabel('Similarity', fontsize=fontsz)
ax.set_title('Similarity btw meas spectrum and sim spectrum', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)

plt.suptitle(f'Matching key: {key_major_name["tone_name"]} Major / {key_minor_name["tone_name"]} minor', fontsize=1.5*fontsz, fontweight="bold")
plt.tight_layout()
plt.show()
