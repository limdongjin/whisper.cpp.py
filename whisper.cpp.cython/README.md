# whisper.cpp.cython

## Installation

```bash
git clone https://github.com/limdongjin/whisper.cpp.py
cd whisper.cpp.py/whisper.cpp.cython

pip install . 
# OR .venv/bin/python -m pip install

cd whisper.cpp
make tiny

cd ..
```

## Usage

```python3
import scipy
from whisper_cpp_py import WhisperContextWrapper, WhisperFullParamsWrapper, transcribe, extract_output

wav_file_path = "examples/jfk.wav"
your_model_file_path = "whisper.cpp/models/ggml-tiny.bin"

context_wrapper = WhisperContextWrapper(absolute_file_path = your_model_file_path)
params_wrapper = WhisperFullParamsWrapper()
    
sr, audio = scipy.io.wavfile.read(wav_file_path)
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
```

## Examples

```
# example1
python examples/example.py

# example2
python examples/example2.py examples/jfk.wav whisper.cpp/models/ggml-tiny.bin
```


