import scipy

def read_audio(filename, sr = 16000):
    if sr != 16000:
        raise Exception("Supported Sample Rate = 16000")

    sr, d =scipy.io.wavfile.read(filename)
    assert sr == 16000

    d = d.astype('float32')/32768.0
    return d
