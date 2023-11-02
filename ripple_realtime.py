"""
====================================
Real-time Visualization Using Ripple
====================================

Procedure:

1. Get channel names, sampling rate
2. Get raw data for 5 - 10 events
  - Stream raw data with 3 s buffer (30 s visualization)
  - On every event, take buffer data +/- 1 s
3. Laplacian re-reference
4. High gamma filter / Welch's method PSD / PCA + SVM classifier (implement later)
5. Compute R2 map


Authors: Alex Rockhill <rockhill@ohsu.edu>

License: BSD-3-Clause
"""
import os.path as op
from joblib import Parallel, delayed
import numpy as np
from scipy import signal, stats
import pandas as pd

import matplotlib.pyplot as plt

import xipppy as xp
import time

data_dir = '.'
bin_size = 5
l_freqs_hga = np.concatenate([np.arange(70, 111, bin_size)])
#                             np.arange(125, 151, bin_size)])


def laplacian_reference(ch_names, data, verbose=True):
    """Reference raw data laplacian.

    Parameters
    ----------
    data : np.ndarray
        The raw data.

    Returns
    -------
    data : np.ndarray
        The re-referenced data.
    """
    raw_data = data
    data = np.zeros_like(raw_data)
    for i, ch in enumerate(ch_names):
        elec_name = "".join([letter for letter in ch if not letter.isdigit()]).rstrip()
        number = "".join([letter for letter in ch if letter.isdigit()]).rstrip()
        j = 0
        pair1 = None
        while j < len(ch_names) and pair1 not in ch_names:
            j += 1
            pair1 = f"{elec_name}{int(number) + j}"
        k = 0
        pair2 = None
        while k > -len(ch_names) and pair2 not in ch_names:
            k -= 1
            pair2 = f"{elec_name}{int(number) + k}"
        if pair1 in ch_names and pair2 in ch_names:
            data[i] = raw_data[i] - (raw_data[ch_names.index(pair1)] +
                                     raw_data[ch_names.index(pair2)]) / 2
            if verbose:
                print(f"Laplacian referencing {ch} to {pair1} and {pair2}")
        elif pair1 in ch_names:
            data[i] = raw_data[i] - raw_data[ch_names.index(pair1)]
            if verbose:
                print(f"Bipolar referencing {ch} to {pair1}")
        elif pair2 in ch_names:
            data[i] = raw_data[ch_names.index(pair2)] - raw_data[i]
            if verbose:
                print(f"Bipolar referencing {pair2} to {ch}")
    return data


def firf(x, f_range, fs=512, w=4):
    """Filter signal with an FIR filter

    *Like fir1 in MATLAB
    x : array-like
        Time series to filter.
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter.
    fs : float, Hz
        The sampling rate.
    w : float
        Length of the filter in terms of the number of cycles
        of the oscillation whose frequency is the low cutoff of the
        bandpass filter/
    Returns
    -------
    x_filt : array-like
        Filtered time series.
    """

    if w <= 0:
        raise ValueError("Number of cycles in a filter must be a positive number.")

    nyq = float(fs / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError("Filter frequencies must be below nyquist rate.")

    if np.any(np.array(f_range) < 0):
        raise ValueError("Filter frequencies must be positive.")

    Ntaps = int(np.floor(w * fs / f_range[0]))
    if x.shape[-1] < Ntaps:
        raise RuntimeError(
            "Length of filter is loger than data. "
            "Provide more data or a shorter filter."
        )

    # Perform filtering
    tapsx = signal.firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = signal.filtfilt(tapsx, [1], x, axis=-1)

    if np.isnan(x_filt).any():
        raise RuntimeError("Filtered signal contains nans. Adjust filter parameters.")

    return x_filt


def compute_activation(data1, data2):
    data = np.concatenate([data1, data2], axis=0)
    n1, n2 = data1.shape[0], data2.shape[0]
    ratio = n1 * n2 / (n1 + n2) ** 2
    mean_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)
    activation = mean_diff**3 / abs(mean_diff) / np.var(data, axis=0) * ratio
    return activation


def get_ch_names():
    ch_names = list()
    while True:
        elec = input('Electrode? (press enter when finished)\t').strip()
        if not elec:
            break
        n_contacts = int(input('Number of contacts?\t').strip())
        for i in range(n_contacts):
            ch_names.append(f'{elec}{i}')
    return ch_names


def compute_hga_freq(epoch, l_freq):
    hga_data = 10 * np.log10(np.abs(signal.hilbert(firf(
        epoch.copy(), (l_freq, l_freq + 5)))))
    hga_data -= hga_data.mean(axis=1, keepdims=True)
    return hga_data


def compute_hga(epoch):
    parallel = Parallel(n_jobs=3)
    out = parallel(delayed(compute_hga_freq)(epoch, l_freq) for l_freq in l_freqs_hga)
    return np.mean(out, axis=0)


sfreq = 2000

buffer = 0.1  # seconds
tmax = 30  # display
tmax_buffer = 3
tmin_epo = -1.5
tmax_epo = 1.5
times_epo = np.linspace(tmin_epo, tmax_epo,
                        int((tmax_epo - tmin_epo) * sfreq) + 1)
rng = np.random.default_rng(11)
order = rng.integers(0, 2, 1000, dtype=bool)

if __name__ == '__main__':
    """with xp.xipppy_open():
        fs_clk = 30000
        xp.signal_set(0, 'raw', True)
        elec_0_raw = xp.cont_raw(300, [0], 0)
        t = np.arange(0, 300000 / fs_clk, 1000/fs_clk, dtype=np.float32)
        plt.plot(t, elec_0_raw[0])
        plt.xlabel('Time(ms)')
        plt.title('Raw Signal for electrode 0')
        plt.show()"""
    sub = input('Subject ID?\t').strip()
    if op.isfile(op.join(data_dir, f'sub-{sub}_chs.csv')):
        ch_names = list(pd.read_csv(op.join(data_dir, f'sub-{sub}_chs.csv'))['ch'])
    else:
        ch_names = get_ch_names()
        pd.DataFrame(dict(ch=ch_names)).to_csv(
            op.join(data_dir, f'sub-{sub}_chs.csv'))
    input('Press enter to begin streaming')

    # 5 second circular buffer, 30 kHz clock time
    # pool = Pool(processes=4) TO DO: add asynchronous
    with xp.xipppy_open():

        # get channel indices
        ch_idx = xp.list_elec('macro', 1024)
        assert ch_idx.size == len(ch_names)

        # turn all channels on hi-res (2 kHz sample rate)
        xp.signal_set(ch_idx, 'hi-res', True)
        for ch_i in ch_idx:
            time.sleep(.01)
            if xp.signal(ch_i, 'hi-res') == 1:
                break
            else:
                raise Exception("Timed out waiting for enable to take effect")

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].set_title('Activation')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Channel')
        axes[0].invert_yaxis()
        axes[1].set_title(ch_names[0])
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('HGA (z-scored)')

        im_r2 = axes[0].imshow(np.zeros((len(ch_names), times_epo.size)) * np.nan,
                               aspect='auto', vmin=-1, vmax=1)

        axes[0].set_xticks(np.linspace(0, times_epo.size, 7))
        axes[1].set_xticks(np.linspace(tmin_epo, tmax_epo, 7))
        for ax in axes:
            ax.set_xticklabels(np.linspace(tmin_epo, tmax_epo, 7).round(2))

        axes[0].set_yticks(range(0, len(ch_names), 10))
        axes[0].set_yticklabels(ch_names[::10])

        line1 = axes[1].plot([np.nan], [np.nan], color='blue', label='condition 1')
        err1 = axes[1].fill_between(times_epo, [np.nan], [np.nan],
                                    color='blue', alpha=0.5)
        line2 = axes[1].plot([np.nan], [np.nan], color='red', label='condition 2')
        err2 = axes[1].fill_between(times_epo, [np.nan], [np.nan],
                                    color='red', alpha=0.5)
        axes[1].legend()
        fig.tight_layout()

        epo_idx = 0
        ch_idx = 0

        def update_plots():
            global line1, err1, line2, err2
            sub_order = order[epo_idx:epo_idx + len(hga)]
            hga1 = np.array([hga[j] for j, test in enumerate(sub_order) if test == 0])
            hga2 = np.array([hga[j] for j, test in enumerate(sub_order) if test == 1])
            if sum(sub_order == 0) > 0 and sum(sub_order == 1) > 0:
                im_r2.set_data(compute_activation(hga1, hga2))
            else:
                im_r2.set_data(np.zeros((len(ch_names), times_epo.size)) * np.nan)
            for obj_group in (line1, err1, line2, err2):
                for obj in (obj_group if isinstance(obj_group, list) else [obj_group]):
                    try:
                        obj.remove()
                    except Exception as e:
                        print(e)
            if sum(sub_order == 0) > 0:
                mean1 = hga1[:, ch_idx].mean(axis=0)
                sem1 = stats.sem(hga1[:, ch_idx], axis=0)
                line1 = axes[1].plot(times_epo, mean1, color='blue')
                err1 = axes[1].fill_between(times_epo, mean1 - sem1, mean1 + sem1,
                                            color='blue', alpha=0.5)
            if sum(sub_order == 1) > 0:
                mean2 = hga2[:, ch_idx].mean(axis=0)
                sem2 = stats.sem(hga2[:, ch_idx], axis=0)
                line2 = axes[1].plot(times_epo, mean2, color='red')
                err2 = axes[1].fill_between(times_epo, mean2 - sem2, mean2 + sem2,
                                            color='red', alpha=0.5)
            fig.canvas.draw()
            fig.canvas.flush_events()

        def keypress(event):
            global epochs, hga, epo_idx, ch_idx
            if event.key == 'c':
                epo_idx += len(epochs)
                epochs = list()
                hga = list()
            elif event.key == 'up':
                ch_idx = ch_idx - 1 % len(ch_names)
            elif event.key == 'down':
                ch_idx = ch_idx + 1 % len(ch_names)
            elif event.inaxes == axes[0]:
                ch_idx = int(round(event.ydata))
            axes[1].set_title(ch_names[ch_idx])
            update_plots()

        fig.canvas.mpl_connect('key_press_event', keypress)
        fig.canvas.mpl_connect('button_press_event', keypress)
        fig.show()

        epochs = list()
        hga = list()
        t1 = t2 = xp.time()
        while True:
            print(t1 - t2)
            # time.sleep(t1 - t2 + 3)
            time.sleep((t1 - t2) / 30000 + 3)
            t1 = xp.time()
            # get event data
            # n, next_events = xp.digin(1024)
            # assert n < 1024
            next_events = [1, 2] if len(epochs) < 10 else []
            for event in next_events[::2]:
                """epochs.append(laplacian_reference(
                    ch_names, rng.random((len(ch_names), times_epo.size))))"""
                epochs.append(laplacian_reference(
                    ch_names,
                    xp.cont_hires(
                        times_epo.size,
                        ch_idx,
                        event.timestamp + int(30000 * tmin_epo)
                    ).reshape(len(ch_names), -1)))
                # compute narrowband hilbert
                hga.append(compute_hga(epochs[-1]))
                print(f'Event at {event.timestamp}')

            # update r2 map and hga plot
            update_plots()

            t2 = xp.time()
