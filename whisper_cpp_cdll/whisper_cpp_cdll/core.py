from scipy.io import wavfile
import ctypes
from .types import WhisperContext, WhisperFullParams, WhisperTokenData
import numpy as np
import math

def init_whisper_and_ctx(libname, fname_model):
    """Return (whisper, WhisperContext)
       
       Parameters:
           libname: file path of 'libwhisper.so' (eg. '/app/whisper.cpp/libwhisper.so')
           fname_model: model file path (eg, '/app/whisper.cpp/models/ggml-tiny.bin')
    """
    whisper = ctypes.CDLL(libname)

    whisper.whisper_init_from_file.restype = ctypes.POINTER(WhisperContext)
    # whisper.whisper_init_from_file.restype = ctypes.c_void_p
    whisper.whisper_full_default_params.restype   = WhisperFullParams
    whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p
    whisper.whisper_full_get_segment_t0.restype = ctypes.c_int
    whisper.whisper_full_get_segment_t1.restype = ctypes.c_int
    whisper.whisper_full_get_token_data.restype = WhisperTokenData
    whisper.whisper_full_get_token_text.restype = ctypes.c_char_p
    whisper.whisper_full.argtypes = [ctypes.POINTER(WhisperContext), WhisperFullParams, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    whisper.whisper_full.restype = ctypes.c_int
    ctx_p = whisper.whisper_init_from_file(fname_model.encode("utf-8"))

    return whisper, ctx_p

def _execute_whisper_full(
    data = None, 
    whisper = None,
    ctx_p = None,
    verbose = False,
    language = b'en',
    n_threads = 4,
    print_realtime = False,
    print_progress = False,
    print_timestamps = False,
    token_timestamps = False,
    suppress_non_speech_tokens = True,
    temperature = 0.0,
    max_len = 30,
    max_tokens = 10,
    beam_search_beam_size = 10,
    greedy_best_of = -1,
    speed_up = False,
    length_penalty = -1.0,
    entropy_thold = 2.4,
    logprob_thold = -1.0,
    no_speech_thold = 0.6
):
    assert data is not None
    assert whisper is not None and ctx_p is not None  
    assert type(data) is not str
    assert type(language) is bytes

    params = whisper.whisper_full_default_params(0)

    params.language = language
    params.n_threads = n_threads
    params.print_realtime = print_realtime 
    params.print_progress = print_progress
    params.print_timestamps = print_timestamps
    params.token_timestamps = token_timestamps
    params.suppress_non_speech_tokens = suppress_non_speech_tokens
    params.temperature = temperature
    params.max_len = max_len
    params.max_tokens = max_tokens
    params.beam_search.beam_size = beam_search_beam_size
    params.greedy.best_of = greedy_best_of
    params.speed_up = speed_up
    params.entropy_thold = entropy_thold
    params.logprob_thold = logprob_thold
    params.no_speech_thold = no_speech_thold
    params.length_penalty = length_penalty
    
    result = whisper.whisper_full(ctx_p, params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))
    if result != 0:
        raise Exception("whisper_full exception!!!")
    if verbose:
        whisper.whisper_print_timings(ctx_p)
    n_seg = whisper.whisper_full_n_segments(ctx_p)
    
    ret = []
    for seg_i in range(n_seg):
        el = {}
        el['segment_id'] = seg_i
        el['text'] = whisper.whisper_full_get_segment_text(ctx_p, seg_i).decode('utf-8', errors='replace')
        el['start'] = int((whisper.whisper_full_get_segment_t0(ctx_p, seg_i) / 100) * 16000.0)
        el['end'] = int((whisper.whisper_full_get_segment_t1(ctx_p, seg_i) / 100) * 16000.0)
        if el['end'] - el['start'] <= 10:
            continue

        el['tokens'] = []
        n_token = whisper.whisper_full_n_tokens(ctx_p, seg_i)
        for token_t in range(n_token):
            token = {}
            token['text'] = whisper.whisper_full_get_token_text(ctx_p, seg_i, token_t).decode('utf-8', errors='replace')
            
            d = whisper.whisper_full_get_token_data(ctx_p, seg_i, token_t)
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
    WINDOW_SIZE = 16000 * 30,
    verbose = False,
    language = b'en',
    n_threads = 4,
    print_realtime = False,
    print_progress = False,
    print_timestamps = False,
    token_timestamps = False,
    suppress_non_speech_tokens = True,
    temperature = 0.0,
    max_len = 30,
    max_tokens = 10,
    beam_search_beam_size = 10,
    greedy_best_of = -1,
    speed_up = False,
    length_penalty = -1.0,
    entropy_thold = 2.4,
    logprob_thold = -1.0,
    no_speech_thold = 0.6
):
    """Run whisper and Return list of segment dict. 

       Parameters:
           data: normalized numpy 1d-array (if you use scipy, then you must do 'wf.astype('float32')/32768.0'.
           libname: file path of 'libwhisper.so' (eg. '/app/whisper.cpp/libwhisper.so')
           fname_model: model file path (eg, '/app/whisper.cpp/models/ggml-tiny.bin')
           WINDOW_SIZE: execution unit size for handle large input. (default: 16000 * 60)

    """

    # whisper, ctx = init_whisper_and_ctx(libname = libname, fname_model = fname_model)

    total_length = data.shape[0]
    spokens = []
    chunks = np.array_split(data, math.ceil(total_length/WINDOW_SIZE))
    start = 0
    for chunk in chunks: 
        whisper, ctx_p = init_whisper_and_ctx(libname = libname, fname_model = fname_model)
        res = _execute_whisper_full(
                data = chunk,
                whisper = whisper, 
                ctx_p = ctx_p,
                verbose = verbose,
                language = language,
                n_threads = n_threads,
                print_realtime = print_realtime,
                print_progress = print_progress,
                print_timestamps = print_timestamps,
                token_timestamps = token_timestamps,
                suppress_non_speech_tokens = suppress_non_speech_tokens,
                temperature = temperature,
                max_len = max_len,
                max_tokens = max_tokens,
                beam_search_beam_size = beam_search_beam_size,
                greedy_best_of = greedy_best_of,
                speed_up = speed_up,
                length_penalty = length_penalty,
                entropy_thold = entropy_thold,
                logprob_thold = logprob_thold,
                no_speech_thold = no_speech_thold
            )
        for segment in res:
            segment['start'] += start
            segment['end'] += start
        start += chunk.size
        spokens.extend(res)
        
        whisper.whisper_free(ctx_p)
        del whisper
        del ctx_p

    return spokens

