import scipy
import pathlib
from whisper_cpp_py import WhisperContextWrapper, WhisperFullParamsWrapper, transcribe, extract_output

if __name__ == "__main__":
    filename = pathlib.Path(__file__).parent.resolve() / "jfk.wav"

    context_wrapper = WhisperContextWrapper(model_name = 'tiny')
    params_wrapper = WhisperFullParamsWrapper()
     
    sr, audio = scipy.io.wavfile.read(filename)
    assert sr == 16000
    
    audio = audio.astype('float32') / 32768.0

    res = transcribe(
            waveform = audio,
            whisper_context_wrapper = context_wrapper,
            whisper_params_wrapper = params_wrapper,
            language = b'en'
        )
    assert res == 0
    res = extract_output(whisper_context_wrapper = context_wrapper)

    print(res)
    
    context_wrapper.destroy_current_context()
    del context_wrapper
    del params_wrapper

