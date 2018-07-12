import numpy as np
import librosa
import librosa.display
import os
import warnings
warnings.filterwarnings("error")


padto = 50000


def mel_spec(path, count_retry=0):
    if count_retry > 5:
        raise IOError("真的是有毒啊 {}".format(path))

    try:
        y, sr = librosa.load(path)
    except:
        file = os.path.basename(path)
        path = os.path.dirname(path)
        file = file[:15] + "{:02}".format(np.random.randint(1, 21)) + file[17:]
        return mel_spec(os.path.join(path, file), count_retry + 1)

    l = len(y)
    left_pad = (padto - l) // 2
    righ_pad = padto - l - left_pad
    y = np.pad(y, (left_pad, righ_pad), 'wrap')

    # librosa.display.waveplot(y, sr=sr)

    # pre_shape = y.shape

    feat = librosa.stft(y, hop_length=512, n_fft=1024)
    feat = np.abs(feat) ** 2
    try:
        feat = librosa.feature.melspectrogram(S=feat, sr=sr, n_mels=128)
    except:
        raise IOError(path)
    #     import ipdb
    #     ipdb.set_tracSe()
    feat = librosa.power_to_db(feat, ref=np.max)
    # print(pre_shape, feat.shape)
    return feat

def specshow(feat):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feat,
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()

