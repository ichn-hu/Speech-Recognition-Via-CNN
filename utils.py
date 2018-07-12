r"""
This file is created for the purpose of DSP project, by Zhifeng Hu
"""
import array
import warnings
import copy
import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import os
from matplotlib import pyplot as plt

from wave import open as open_wave


data_dir = "/home/zfhu/playground/DSP/data/"


class DataFeed(object):
    def __init__(self):
        self.data_dir = data_dir
        self.stuids = os.listdir(self.data_dir)
        self.cates = "语音 余音 识别 失败 中国 忠告 北京 背景 上海 商行 复旦 饭店 Speech Speaker Signal File Print Open Close Project".split(
            ' ')

    def __len__(self):
        return len(self.stuids)

    def get_path(self, stu, cate, ith):
        assert 0 <= stu < len(self)
        assert 0 <= cate < 20
        assert 0 <= ith < 20
        stuid = self.stuids[stu]
        ret = os.path.join(self.data_dir, stuid, "{0}-{1:02}-{2:02}.wav".format(stuid, cate, ith + 2))
        return ret

    def get_blob(self, stu, cate, ith):
        path = self.get_path(stu, cate, ith)
        return open(path, "rb").read()

    def get_by_id(self, num):
        ith = num % 20
        num //= 20
        cate = num % 20
        num //= 20
        assert num < len(self)
        return self.get_path(num, cate, ith), cate


def normalize(ys, amp=1.0):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.

    ys: wave array
    amp: max amplitude (pos or neg) in result

    returns: wave array
    """
    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)


class Wave:

    def __init__(self, ys, ts=None, framerate=None):
        """Initializes the wave.

        ys: wave array
        ts: array of times
        framerate: samples per second
        """
        self.ys = np.asanyarray(ys)
        self.framerate = framerate if framerate is not None else 11025

        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)

    def copy(self):
        """Makes a copy.

        Returns: new Wave
        """
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.ys)

    def downsampling(self, new_framerate=4000):
        """
        48000 frame/second 的采样率明显太高，我们应该考虑降采样来提高效率
        :return:
        """
        raise NotImplemented

    def get_short_time_energy(self, seg_length):
        window = np.hamming(seg_length)
        i, j = 0, seg_length
        step = seg_length // 2

        # map from time to Spectrum
        ste = []
        x = []

        while j < len(self.ys):
            segment = self.slice(i, j)
            segment.window(window)

            # the nominal time for this segment is the midpoint
            t = (segment.start + segment.end) / 2
            x.append(t)
            ste.append(np.sum(segment.ys * segment.ys))

            i += step
            j += step

        return Wave(ste, ts=x, framerate=1 / (x[1] - x[0]))

    def get_short_time_cross_rate(self, seg_length):
        i, j = 0, seg_length
        step = seg_length // 2

        # map from time to Spectrum
        ste = []
        x = []

        while j < len(self.ys):
            segment = self.slice(i, j)
            cross_rate = np.sum(np.abs(np.diff(np.abs(segment.ys)))) / 2 / seg_length

            # the nominal time for this segment is the midpoint
            t = (segment.start + segment.end) / 2
            x.append(t)
            ste.append(cross_rate)

            i += step
            j += step

        return Wave(ste, ts=x, framerate=1 / (x[1] - x[0]))

    def plot_short_time_feature(self, seg_length):
        ste = self.get_short_time_energy(seg_length)
        stc = self.get_short_time_cross_rate(seg_length)
        fig, (w, e, c) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        w.plot(self.ts, self.ys)
        w.set_title(u"波形")
        e.plot(ste.ts, ste.ys)
        e.set_title(u"短时能量")
        c.plot(stc.ts, stc.ys)
        c.set_title(u"短时平均过零率")
        # fig.show()
        plt.show()

    def endian_detection(self, plot=False, w_axe=None, e_axe=None, c_axe=None):
        frame_length = 0.05
        seg_lenght = int(frame_length * self.framerate) + 1
        ste = self.get_short_time_energy(seg_lenght)
        stc = self.get_short_time_cross_rate(seg_lenght)
        e_t = 1
        c_t = 0.002

        def search(es, cs):
            t = len(es)
            p = t
            for i in range(t):
                if es[i] > e_t:
                    p = i
                    break
            pp = p
            for i in range(p):
                if cs[i] > c_t:
                    pp = i
                    break
            return pp

        b = search(ste.ys, stc.ys)
        e = search(ste.ys[::-1], stc.ys[::-1])
        b, e = ste.ts[b], ste.ts[::-1][e]

        def search_index(ts, p):
            for i in range(len(ts)):
                if ts[i] > p:
                    return i
            return -1

        begin = search_index(self.ts, b)
        end = search_index(self.ts, e)

        if plot:
            fig, (w_axe, e_axe, c_axe) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            w_axe.plot(self.ts, self.ys)
            w_axe.set_title(u"波形")
            w_axe.plot(self.ts[begin:end], self.ys[begin:end], 'r')
            e_axe.plot(ste.ts, ste.ys)
            e_axe.set_title(u"短时能量")
            c_axe.plot(stc.ts, stc.ys)
            c_axe.set_title(u"短时平均过零率")
            plt.show()

    def pre_emphasis(self, alpha=0.97):
        self.ys[1:] = self.ys[1:] - alpha * self.ys[:-1]

    def mfcc(self):
        import numpy
        signal = self.ys
        pre_emphasis = 0.97
        frame_size = 0.025
        sample_rate = self.framerate
        frame_stride = 0.01
        nfilt = 40
        NFFT = 1024
        num_ceps = 12
        cep_lifter = 22

        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(
            float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal,
                                  z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
            numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]

        frames *= numpy.hamming(frame_length)

        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale

        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)

        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB

        mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

        (nframes, ncoeff) = mfcc.shape
        n = numpy.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
        mfcc *= lift  # *
        return filter_banks, mfcc

    def extract_mel_feature(self, frame_size=1024, overlap=0.5, nmel=64, nceps=12, normalize=True):
        self.pre_emphasis()
        wave_ys, wave_framerate = self.ys, self.framerate
        # 分帧
        samples = len(wave_ys)
        step_size = int(round(frame_size * overlap))
        n_frames = samples // step_size
        frames = np.hstack(
            [wave_ys[i: i + frame_size].reshape(-1, 1) for i in range(0, samples - frame_size, step_size)])
        print(frames.shape, step_size, n_frames)

        # 加hamming窗
        hamming_window = np.hamming(frame_size)
        frames = frames * hamming_window.reshape(-1, 1)

        # 离散傅里叶变换 + 能量谱
        frames = np.abs(np.fft.rfft(frames)) ** 2

        # 加mel滤波器
        max_mel_freq = (2595 * np.log10(1 + wave_framerate / 2 / 700))
        min_mel_freq = 0
        mel_freqs = np.linspace(min_mel_freq, max_mel_freq, nmel + 2)
        hz_freqs = (700 * (10 ** (mel_freqs / 2595) - 1))
        # fft_freqs = np.fft.rfftfreq(frame_size, 1 / wave_framerate)
        freq_idx = np.floor((frame_size + 1) / wave_framerate * hz_freqs).astype(int)
        mel_filters = np.zeros([nmel, frame_size])
        for i in range(1, nmel + 1):
            l = freq_idx[i - 1]
            m = freq_idx[i]
            r = freq_idx[i + 1]
            for k in range(l, m):
                mel_filters[i - 1][k] = (k - l) / (m - l)
            for k in range(m, r):
                mel_filters[i - 1][k] = (r - k) / (r - m)

        melfreqfeat = mel_filters.dot(frames)
        mfcc = scipy.fftpack.dct(melfreqfeat, type=2, axis=0, norm='ortho')[1: (nceps + 1), :]
        # todo: lifter
        # todo: to db
        if normalize:
            melfreqfeat -= np.mean(melfreqfeat, 1, keepdims=True)
        mfcc -= np.mean(mfcc, 1, keepdims=True)
        return melfreqfeat, mfcc


def play_wave(filename='sound.wav'):
    """Plays a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    # cmd = 'powershell -c (New-Object Media.SoundPlayer "{0}").PlaySync();'.format(filename)
    # cmd = 'start "{0}"'.format(filename)
    # print(cmd)
    # popen = subprocess.Popen(cmd, shell=True)
    # popen.communicate()
    import winsound
    winsound.PlaySound(filename, winsound.SND_FILENAME)


def read_wave(filename='sound.wav'):
    fp = open_wave(filename, 'r')

    nchannels = fp.getnchannels()
    nframes = fp.getnframes()
    sampwidth = fp.getsampwidth()
    framerate = fp.getframerate()

    z_str = fp.readframes(nframes)

    fp.close()

    dtype_map = {1: np.int8, 2: np.int16, 3: 'special', 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError('sampwidth %d unknown' % sampwidth)

    if sampwidth == 3:
        xs = np.fromstring(z_str, dtype=np.int8).astype(np.int32)
        ys = (xs[2::3] * 256 + xs[1::3]) * 256 + xs[0::3]
    else:
        ys = np.fromstring(z_str, dtype=dtype_map[sampwidth])

    # if it's in stereo, just pull out the first channel
    if nchannels == 2:
        ys = ys[::2]

    # ts = np.arange(len(ys)) / framerate
    wave = Wave(ys, framerate=framerate)
    wave.ys = normalize(wave.ys)
    return wave


def plot_wave(filename):
    wave = read_wave(filename)
    plt.plot(wave.ts, wave.ys)
    print(wave.framerate)
    # plt.show()


def plot_spectrogram(filename):
    wave = read_wave(filename)
    spec = wave.make_spectrogram(1024)
    spec.plot()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data = DataFeed()
    a, __ = data.get_by_id(100)
    w = read_wave(a)
    play_wave(a)
    import time

    start = time.time()
    print(start)
    a, b = w.mfcc()
    print(time.time() - start)
    plt.imshow(a.T)
    plt.show()
    print(w.framerate)
    # plot_spectrogram(w)
    # np.random.seed(0)
    # for i in range(10):
    #     a, __ = data.get_by_id(np.random.randint(0, 32 * 20 * 20))
    #     wave = read_wave(a)
    #     wave.endian_detection(plot=True)
    # play_wav(a)
