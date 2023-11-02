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
from scipy import signal
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

    # compute the activation
    activation = np.zeros((data.shape[1])) * np.nan
    for i in range(data1.shape[1]):
        mean_diff = np.mean(data1[:, i]) - np.mean(data2[:, i])
        activation[i] = (
            mean_diff**3 / abs(mean_diff) / np.var(data[:, i]) * ratio
        )
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


def compute_hga(epoch):
    hga = np.zeros_like(epoch)
    for l_freq in l_freqs_hga:
        hga_data = 10 * np.log10(np.abs(signal.hilbert(firf(
            epoch.copy(), (l_freq, l_freq + 5)))))
        hga_data -= hga_data.mean(axis=1, keepdims=True)
        hga += hga_data
    hga /= l_freqs_hga.size
    return hga


# hardware sampling frequency, downsampled frequency
sfreq = 30000
sfreq2 = 2000

buffer = 0.1  # seconds
tmax = 30  # display
tmax_buffer = 3
tmin_epo = -1.5
tmax_epo = 1.5
times_epo = np.linspace(tmin_epo, tmax_epo,
                        int((tmax_epo - tmin_epo) * sfreq2) + 1)

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
    ch_idx = xp.list_elec('macro', 1024)
    assert ch_idx.size == len(ch_names)

    input('Press enter to begin streaming')

    with xp.xipppy_open():
        xp.signal_set(ch_idx, 'raw', True)
        for _ in range(100):
            time.sleep(.01)
            if xp.signal(ch_idx, 'raw') == 1:
                break
            else:
                raise Exception("Timed out waiting for enable to take effect")

        while True:
            batch = input('Event batch size (enter if done)?\t').strip()
            if not batch:
                break
            batch = int(batch)

            epochs = np.zeros((batch, len(ch_names),
                               (tmax_epo - tmin_epo) * sfreq2 + 1))
            events = list()
            event_ids = list()
            epo_idx = 0

            last_data_buffer = None
            data_buffer = np.zeros((len(ch_names), int(sfreq2 * tmax_buffer)))
            b_idx = 0
            last_time_stamps = None
            time_stamps = np.zeros((int(sfreq2 * tmax_buffer)))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(r'Voltage ($\mu$V)')
            ax.set_xlim([0, tmax])
            fig.show()

            traces = [list() for _ in range(101)]
            t_idx = 0

            while len(epochs) < batch:

                if t_idx > tmax / buffer - buffer:
                    t_idx = 0
                    ax.cla()
                    ax.set_xlim([0, 10])

                if b_idx > tmax_buffer / buffer - buffer:
                    # process events from last buffer (might use this buffer)
                    for event in events.copy():
                        data_buffer_all = np.concatenate(
                            [last_data_buffer, data_buffer], axis=-1)
                        time_stamps_buffer = np.arange(last_time_stamps[0],
                                                       time_stamps[-1])
                        # must be before second half of this buffer (save for next)
                        if event < time_stamps[int(tmax_buffer / buffer / 2)]:
                            event_idx = np.argmin(abs(time_stamps_buffer - event))
                            slice_event = slice(int(event_idx + tmin_epo * sfreq2),
                                                int(event_idx + tmax_epo * sfreq2))
                            epochs[epo_idx] = data_buffer_all[:, slice_event]
                            epo_idx += 1
                            events.remove(event)
                    last_buffer_data = data_buffer
                    last_time_stamps = time_stamps

                for line in traces[t_idx]:
                    line.remove()

                # get event data
                n, events2 = xp.digin(1024)
                assert n < 1024
                for event in events2:
                    events.append(event.timestamp)
                    event_ids.append(event.sma1)
                    print(f'Event {event.sma1} recieved at {event.timestamp}')

                # get raw data
                t = t_idx * buffer
                data, ts = xp.cont_raw(int(sfreq * buffer), ch_idx, 0).reshape(
                    len(ch_names), -1)
                time_stamps[b_idx] = ts
                # data = rng.random((len(ch_names), int(sfreq * buffer))) * 1e-4 + \
                #    np.sin(np.arange(t, t + buffer, int(sfreq * buffer)) / sfreq)[:, None]
                idx_buffer = int(t_idx * buffer * sfreq2)
                data_buffer[:, idx_buffer:idx_buffer + int(sfreq2 * buffer)] = \
                    data[:, ::sfreq // sfreq2]

                # plot raw trace
                traces[t_idx] = ax.plot(
                    np.linspace(t, t + buffer, data.shape[1], dtype=np.float32),
                    (data + 4e-3 * np.arange(len(ch_names))[:, None]).T,
                    color='black', scalex=False)
                fig.canvas.draw()
                fig.canvas.flush_events()
                t_idx += 1
                b_idx += 1

            # plot evoked
            event_ids = np.array(event_ids)
            ids = np.unique(event_ids)
            fig, axes = plt.subplots(len(ids), 1, figsize=(8, 8))
            for ax, event_id in zip(axes, ids):
                ax.set_title(event_id)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(r'Voltage ($\mu$V)')
                ax.plot(times_epo, epochs[event_ids == event_id].mean(axis=0).T)
            fig.tight_layout()
            fig.show()

            # laplacian re-reference
            epochs = laplacian_reference(ch_names, epochs)

            # compute narrowband hilbert
            hga = np.array(Parallel(n_jobs=4)(delayed(compute_hga)(epoch)
                                              for epoch in epochs))

            # plot hga
            fig, axes = plt.subplots(len(ids), 1, figsize=(8, 8))
            for ax, event_id in zip(axes, ids):
                ax.set_title(event_id)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('HGA (dB)')
                ax.plot(times_epo, hga[event_ids == event_id].mean(axis=0).T)
            fig.tight_layout()
            fig.show()

            # plot r2 map
            r2_map = np.zeros((len(ch_names), times_epo.size))
            for i in range(len(ch_names)):
                r2_map[i] = compute_activation(hga[event_ids == ids[0], i],
                                               hga[event_ids == ids[1], i])

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.invert_yaxis()
            ax.imshow(r2_map, aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(np.linspace(0, times_epo.size, 7))
            ax.set_xticklabels(np.linspace(tmin_epo, tmax_epo, 7).round(2))
            ax.set_yticks(np.arange(len(ch_names)))
            ax.set_yticklabels(ch_names)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Activation')
            fig.tight_layout()
            fig.show()
