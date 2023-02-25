from scipy.io import wavfile
import ctypes
from .types import WhisperFullParams, WhisperTokenData

def init_whisper_and_ctx(libname, fname_model):
    """Return (whisper, WhisperContext)
       
       Parameters:
           libname: file path of 'libwhisper.so' (eg. '/app/whisper.cpp/libwhisper.so')
           fname_model: model file path (eg, '/app/whisper.cpp/models/ggml-tiny.bin')
    """
    whisper = ctypes.CDLL(libname)
    whisper.whisper_init_from_file.restype = ctypes.c_void_p
    whisper.whisper_full_default_params.restype   = WhisperFullParams
    whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p
    whisper.whisper_full_get_segment_t0.restype = ctypes.c_int
    whisper.whisper_full_get_segment_t1.restype = ctypes.c_int
    whisper.whisper_full_get_token_data.restype = WhisperTokenData
    whisper.whisper_full_get_token_text.restype = ctypes.c_char_p
    ctx = whisper.whisper_init_from_file(fname_model.encode("utf-8"))

    return whisper, ctx

def _execute_whisper_full(
    data = None, 
    whisper = None,
    ctx = None,
    language = b'en',
    n_threads = 4,
    print_realtime = False,
    print_progress = False,
    suppress_non_speech_tokens = True
):
    assert data is not None
    assert whisper is not None and ctx is not None  
    assert type(data) is not str
    assert type(language) is bytes

    params = whisper.whisper_full_default_params(0)
    params.n_threads = n_threads
    params.print_realtime = print_realtime 
    params.print_progress = print_progress
    params.suppress_non_speech_tokens = suppress_non_speech_tokens
    params.language = language

    result = whisper.whisper_full(ctypes.c_void_p(ctx), params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))
    n_seg = whisper.whisper_full_n_segments(ctypes.c_void_p(ctx))

    ret = []
    for seg_i in range(n_seg):
        el = {}
        el['segment_id'] = seg_i
        el['text'] = whisper.whisper_full_get_segment_text(ctypes.c_void_p(ctx), seg_i).decode('utf-8', errors='replace')
        el['start'] = int((whisper.whisper_full_get_segment_t0(ctypes.c_void_p(ctx), seg_i) / 100) * 16000.0)
        el['end'] = int((whisper.whisper_full_get_segment_t1(ctypes.c_void_p(ctx), seg_i) / 100) * 16000.0)
        if el['end'] - el['start'] <= 10:
            continue

        el['tokens'] = []
        n_token = whisper.whisper_full_n_tokens(ctypes.c_void_p(ctx), seg_i)
        for token_t in range(n_token):
            token = {}
            token['text'] = whisper.whisper_full_get_token_text(ctypes.c_void_p(ctx), seg_i, token_t).decode('utf-8', errors='replace')
            
            d = whisper.whisper_full_get_token_data(ctypes.c_void_p(ctx), seg_i, token_t)
            token['id'] = d.id
            token['tid'] = d.tid
            token['p'] = d.p
            token['plog'] = d.plog
            token['pt'] = d.pt
            token['ptsum'] = d.ptsum
            token['t0'] = d.t0
            token['t1'] = d.t1
            token['vlen'] = d.vlen

            del d
            el['tokens'].append(token)

        ret.append(el)

    return ret

def run_whisper(
    data,
    libname, 
    fname_model,
    WINDOW_SIZE = 16000 * 30 * 2,
    language = b'en',
    n_threads = 4,
    print_realtime = False,
    print_progress = False,
    suppress_non_speech_tokens = True
):
    """Run whisper and Return list of segment dict. 

       Parameters:
           data: normalized numpy 1d-array (if you use scipy, then you must do 'wf.astype('float32')/32768.0'.
           libname: file path of 'libwhisper.so' (eg. '/app/whisper.cpp/libwhisper.so')
           fname_model: model file path (eg, '/app/whisper.cpp/models/ggml-tiny.bin')
           WINDOW_SIZE: execution unit size for handle large input. (default: 16000 * 60)

    """
    whisper, ctx = init_whisper_and_ctx(libname = libname, fname_model = fname_model)

    total_length = data.shape[0]
    spokens = []
    for start in range(0, total_length, WINDOW_SIZE):
        res = _execute_whisper_full(
                data = data[start:start+WINDOW_SIZE].copy(), 
                whisper = whisper, 
                ctx = ctx,
                language = language,
                n_threads = n_threads,
                print_realtime = print_realtime,
                print_progress = print_progress,
                suppress_non_speech_tokens = suppress_non_speech_tokens
            )
        for segment in res:
            segment['start'] += start
            segment['end'] += start
        spokens.extend(res)

    whisper.whisper_free(ctypes.c_void_p(ctx))
    del whisper
    del ctx

    return spokens

