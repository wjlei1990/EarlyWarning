from arrival_pick import highpass_filter_waveform


def cut_P_wave(trace, arrival_time, time_padded_before, win_length):
    """
    Cut out the first win_length seconds of P wave 
    (Careful: for stations close to the event, S wave
    may be included.)
    """
    starttime = arrival_time - time_padded_before
    endtime = starttime + win_length
    trace_cut = trace.slice(starttime=starttime, endtime=endtime)
    return trace_cut


def differentiate_waveform(trace):
    """
    Convert velocity to acceleration
    """
    trace.differentiate()
    highpass_filter_waveform(trace, 0.075)
    return trace


def integrate_waveform(trace):
    """
    Convert acceleration to velocity,
    or velocity to displacement
    """
    trace.integrate()
    highpass_filter_waveform(trace, 0.075)
    return trace


def low_filter_waveform(trace, freq):
    trace.detrend('linear')
    trace.taper(0.1)
    trace.filter('lowpass', freq=freq, corners=4, zerophase=False)
    trace.detrend('linear')
    trace.taper(0.1)


def make_disp_vel_acc_records(trace):
    """
    Several empircal measurements for earthquake early warning 
    from literature (tau_p_max, tau_c, P_d, P_v, P_a)
    """
    #    trace_len = len(trace)
    station_type = trace.stats.channel[:2]
    if (station_type == 'BH') | (station_type == 'HH'):
        trace_vel = trace.copy()
        trace_disp = integrate_waveform(trace_vel.copy())
        trace_acc = differentiate_waveform(trace_vel.copy())
    elif (station_type == 'HN') | (station_type == 'HL'):
        trace_acc = trace.copy()
        trace_vel = integrate_waveform(trace_acc.copy())
        trace_disp = integrate_waveform(trace_vel.copy())
    else:
        raise ValueError('Wrong instrument! Choose from BH*, HH*, HN*, or HL*.')
    return trace_acc, trace_vel, trace_disp


def tau_p_max(trace, df, time_padded_before,
              lowpass_for_tau_p=True):
    """
    tau_p_i = 2 * pi * sqrt(X_i/D_i)
    X_i = alpha * X_(i-1) + x_i^2
    D_i = alpha * D_(i-1) + ((dx/dt)_i)^2
    x_i is velocity. 
    Olson and Allen (2005) used 3 Hz lowpass filter.
    Previous studies show that this measurement seems to be not very robust.
    """
    trace_len = len(trace)
    if lowpass_for_tau_p:
        low_filter_waveform(trace, 3)

    trace_acc, trace_vel, _ = make_disp_vel_acc_records(trace)
    alpha = 1 - 1 / df
    tau_p = np.zeros(trace_len, )
    X_i = np.zeros(trace_len, )
    D_i = np.zeros(trace_len, )
    for i_sample in range(trace_len):
        if i_sample == 0:
            X_i[i_sample] = trace_vel[i_sample] ** 2
            D_i[i_sample] = trace_acc[i_sample] ** 2
        else:
            X_i[i_sample] = X_i[i_sample - 1] * alpha + trace_vel[i_sample] ** 2
            D_i[i_sample] = D_i[i_sample - 1] * alpha + trace_acc[i_sample] ** 2
    tau_p = 2 * np.pi * np.sqrt(X_i / D_i)
    tau_p[np.isnan(tau_p)] = 0
    tau_p_max = max(tau_p[int(time_padded_before * df + 1):])

    return tau_p_max, tau_p


def tau_c_and_peak_amplitude(trace):
    """
    tau_c is a period parameter. If the waveform is monochromatic,
    this is essentially the period.
    Pd, Pv, Pa are the peak amplitude of displacement, velcoity,
    acceleration within the first a few seconds.
    """
    trace_acc, trace_vel, trace_disp = make_disp_vel_acc_records(trace)
    tau_c = 2 * np.pi * np.sqrt(np.sum(trace_disp.data ** 2) / np.sum(trace_vel.data ** 2))
    P_d = max(trace_disp)
    P_v = max(trace_vel)
    P_a = max(trace_acc)
    return tau_c, P_d, P_v, P_a