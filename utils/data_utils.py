from scipy.signal import butter, filtfilt


def butterworth_filter(data, cutoff=1, fs=30, order=4):
    """
    The `butterworth_filter` function implements a Butterworth filter on input data with specified
    cutoff frequency, sampling rate, and filter order.

    Arguments:

    * `data`: Data is the input signal that you want to filter using the Butterworth filter. It could be
    a 1D array or a list containing the signal values over time that you want to process.
    * `cutoff`: The `cutoff` parameter in the `butterworth_filter` function represents the cutoff
    frequency of the Butterworth filter. This is the frequency at which the filter starts attenuating
    the input signal. It is specified in hertz (Hz) and defaults to 1 Hz if not provided.
    * `fs`: The `fs` parameter in the `butterworth_filter` function represents the sampling frequency of
    the input data. It is used to calculate the Nyquist frequency, which is half of the sampling
    frequency. This Nyquist frequency is important for normalizing the cutoff frequency when designing
    the Butterworth filter.
    * `order`: The `order` parameter in the `butterworth_filter` function refers to the order of the
    Butterworth filter to be used for filtering the input data. A higher order filter will have a
    steeper roll-off but may introduce more phase distortion. It determines how quickly the filter
    attenuates frequencies beyond

    Returns:

    The function `butterworth_filter` returns the filtered data `y` after applying a Butterworth filter
    to the input data.
    """
    # 计算归一化截止频率
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if normal_cutoff <= 0 or normal_cutoff >= 1:
        raise ValueError(f"截止频率必须在 (0, Nyquist) 范围内, 现在是{normal_cutoff:.2f}")

    # 设计Butterworth滤波器
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # 使用filtfilt函数进行双向滤波，避免相位延迟
    y = filtfilt(b, a, data)
    return y
