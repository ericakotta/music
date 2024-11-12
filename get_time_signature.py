import scipy
import numpy as np
import matplotlib.pyplot as plt
import tools

wav_filename = r"Super Mario Bros. Theme  but its in a minor key.wav"
sample_input = scipy.io.wavfile.read(wav_filename)
sample_rate = sample_input[0]
# Just get a single channel
raw_sample_data = [x[0] for x in sample_input[1]]

dt = 1 / sample_rate
N = len(raw_sample_data)
raw_sample_time = np.arange(N) * dt
max_amp = np.max(raw_sample_data)

# Crop wav file to first x seconds
song_window_seconds = [0.838, 25]
idx_window = [int(song_window_seconds[i] * sample_rate) for i in range(2)]

sample_data = np.array(raw_sample_data[idx_window[0]:idx_window[1]])
sample_time = raw_sample_time[idx_window[0]:idx_window[1]]

dx = np.mean([sample_time[i+1] - sample_time[i] for i in range(len(sample_time)-2)])
fontsz = 8

# Search for a section that repeats
min_div_seconds, max_div_seconds = [5, 7]
div_steps = 3
div_step_seconds = 0.1
scan_step_seconds = 1.0

offset_seconds = np.linspace(2.4, 2.8, 5)
for offset in offset_seconds:
    print(f"OFFSET: {offset}s")

    fig, axs = plt.subplots(5,2)
    axs = axs.flatten()

    ax = axs[0]
    ax.plot(sample_time, sample_data)
    ax.plot([offset]*2, [min(sample_data), max(sample_data)], 'r', linestyle='dashed', label='Offset')
    ax.set_title('Sampled data', fontsize=fontsz)
    ax.set_xlabel('t (s)', fontsize=fontsz)
    ax.set_ylabel('amp', fontsize=fontsz)
    ax.tick_params(axis='both', labelsize=fontsz)

    for div_idx, div_seconds in enumerate(np.linspace(min_div_seconds, max_div_seconds, div_steps)):
        trange = [offset, offset + div_seconds]
        print(f"\nDIV SECONDS: {div_seconds}")
        template_idx = np.where((sample_time >= trange[0]) & (sample_time < trange[-1]))[0]
        template = tools.standardize(sample_data[template_idx])

        ax = axs[div_idx+1]
        ax.plot(template, color='b', alpha=0.4)
        ax.set_title(f"{div_seconds}")

        div_length = int(div_seconds / dx)
        step_length = int(scan_step_seconds / dx)
        idx_starts = range(template_idx[-1], len(sample_data) - div_length, step_length)
        
        nxcorrs = []
        for idx_start in idx_starts:
            idx_end = idx_start + div_length
            shifted_template = tools.standardize(sample_data[idx_start:idx_end])
            xcorr = np.sum(np.multiply(template, shifted_template))
            nxcorr = (1 / len(template)) * (
                np.sum(np.multiply(template, shifted_template))
            ) / (
                np.sqrt(np.sum(template ** 2)) * np.sqrt(np.sum(shifted_template ** 2))
            )
            nxcorrs.append(xcorr)
            if len(template) != len(shifted_template):
                print(f"Len template: {len(template)}, shifted: {len(shifted_template)}")

        best_xcorr_idx = np.argmax(nxcorrs)
        best_template = tools.standardize(sample_data[idx_starts[best_xcorr_idx]:idx_starts[best_xcorr_idx] + div_length])
        print(f"Best nxc: {nxcorrs[best_xcorr_idx]}")
        ax.plot(best_template, alpha=0.4, color='r')
        ax.set_title(f'Best xcorr: {nxcorrs[best_xcorr_idx]}', fontsize=fontsz)

plt.show()


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

