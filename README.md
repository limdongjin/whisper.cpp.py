# whisper.cpp.py

- git clone https://github.com/limdongjin/whisper.cpp.py
- cd whisper.cpp.py
- pip install .
- cd whisper.cpp
- make tiny
- cd ..

```python3
from whisper_cpp_py import WhisperContextWrapper, WhisperFullParamsWrapper, transcribe, extract_output

filename = "./examples/jfk.wav"

context_wrapper = WhisperContextWrapper('tiny')
params_wrapper = WhisperFullParamsWrapper()
    
audio, sr = librosa.load(filename, sr=16000)

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
```
