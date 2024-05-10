import scipy
import numpy as np
import matplotlib.pyplot as plt
import tools

wav_filename = r"C:\Users\erica\OneDrive\Desktop\repos\music\Super Mario Bros. Theme  but its in a minor key.wav"
sample_input = scipy.io.wavfile.read(wav_filename)
sample_rate = sample_input[0]
# Just get a single channel
raw_sample_data = [x[0] for x in sample_input[1]]

dt = 1 / sample_rate
N = len(raw_sample_data)
raw_sample_time = np.arange(N) * dt
max_amp = np.max(raw_sample_data)

# Crop wav file to first x seconds
seconds_window = [0.838, 20]
sample_window_seconds = [0.838, 20]
idx_window = [int(sample_window_seconds[i] * sample_rate) for i in range(2)]

sample_data = raw_sample_data[idx_window[0]:idx_window[1]]
sample_time = raw_sample_time[idx_window[0]:idx_window[1]]

fontsz = 8
fig, axs = plt.subplots(2,2)
axs = axs.flatten()

ax = axs[0]
ax.plot(sample_time, sample_data)
ax.set_title('Sampled data', fontsize=fontsz)
ax.set_xlabel('t (s)', fontsize=fontsz)
ax.set_ylabel('amp', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)

sample_fft = tools.get_fft(sample_time, sample_data)
xf, yf = sample_fft[0], sample_fft[1]
ax = axs[1]
ax.plot(xf, yf.real, 'b', label='real')
ax.plot(xf, yf.imag, 'r', label='imag')
ax.plot(xf, np.abs(yf), 'k', label='abs')
ax.plot(xf, np.abs(yf), 'k', label='abs')
ax.legend(fontsize=fontsz)
ax.set_title('Sampled data spectrum', fontsize=fontsz)
ax.set_xlabel('Hz', fontsize=fontsz)
ax.set_ylabel('amp', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)
ax.set_xlim([-3000, 3000])

see_idx = np.where((xf >= 256) & (xf <= 512))#xf.index(see_x[0]), xf.index(see_x[1])
plt_xf = np.array(xf)[see_idx]
plt_yf = np.array(np.abs(yf))[see_idx]

key_steps = 20
key_tones = [440 * (2 ** (i / 12.)) for i in range(-9, 3)]
middle_tones = [440 * (2 ** (i / 12.)) for i in range(-9, 3)]
key_tones = []
for tone in middle_tones:
    key_tones.extend([tone * (2 ** (i / (12 * key_steps))) for i in range(key_steps)])

meas_yf = (plt_yf - min(plt_yf)) / (max(plt_yf) - min(plt_yf))
conv_sigma = 0.2
meas_yf_conv = tools.convolve_with_gaussian(plt_xf, meas_yf, sigma=conv_sigma)
scores = []
sim_width = 2
for key_tone in key_tones:
    sim_yf = tools.simulate_scale_spectrum(plt_xf, tone=key_tone, width=sim_width)
    score = np.sum(np.multiply(meas_yf_conv, sim_yf)**2)
    scores.append(score)

best_match_idx = scores.index(max(scores))
best_match_tone = key_tones[best_match_idx]
best_match_yf = tools.simulate_scale_spectrum(plt_xf, tone=best_match_tone, width=sim_width)

ax = axs[2]
ax.plot(plt_xf, best_match_yf, 'r', alpha=1, label=f"sim {round(best_match_tone,1)}Hz")
ax.plot(plt_xf, meas_yf, 'k', label=f"meas")
ax.plot(plt_xf, meas_yf_conv, 'g', label=f"meas conv {conv_sigma}Hz")
ax.set_title(f'Spectrum zoom', fontsize=fontsz)
ax.legend(fontsize=fontsz)
ax.set_xlabel('Hz', fontsize=fontsz)
ax.set_ylabel('amp', fontsize=fontsz)
ax.tick_params(axis='both', labelsize=fontsz)

key_major = best_match_tone
key_minor = key_major * (2 ** (-3. / 12.))
key_major_name = tools.name_tone(key_major)
key_minor_name = tools.name_tone(key_minor)
# print(f"{round(key_major,1)} Hz, {key_major_name} (major) / {key_minor_name} (minor)")
# match_key_label = f"{key_major_name.split(' ')[0].upper()} (Major) / {key_minor_name.split(' ')[0].lower()} (minor)"
# axs[3].plot(key_tones, scores, 'ro')

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


# fig, ax = plt.subplots(2)
# sim_yf = simulate_scale_spectrum(plt_xf, tone=228.07, width=3)
# ax[0].plot(plt_xf, sim_yf,'r')
# ax[0].plot(plt_xf, meas_yf, 'k')
# ax[1].plot(key_tones, scores, 'ro')
# plt.show()
