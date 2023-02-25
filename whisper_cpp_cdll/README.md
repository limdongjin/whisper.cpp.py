# whisper.cpp.cdll

## Quick Start

**1. Install whisper.cpp**
```bash
git clone https://github.com/ggerganov/whisper.cpp

cd whisper.cpp
make tiny
make libwhisper.so
```

**2. Install whisper_cpp_cdll**
```bash
pip install whisper_cpp_cdll
```

**3. Usage**
```python3
from whisper_cpp_cdll.core import run_whisper
from whisper_cpp_cdll.util import read_audio

# your whisper.cpp files path
libname = './whisper.cpp/libwhisper.so'
fname_model = './whisper.cpp/models/ggml-tiny.bin'
d = read_audio('./whisper.cpp/samples/jfk.wav')

result = run_whisper(data = d, libname = libname, fname_model = fname_model, language=b'en')
#=> [{'segment_id': 0, 'text': ' And so my fellow Americans ask not what your country can do for you ask what you can do for your country.', 'start': 0, 'end': 176000, 'tokens': [{..}]},..... ]
```
