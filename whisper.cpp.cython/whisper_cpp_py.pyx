#!python
# cython: language_level=3
from libcpp cimport bool
import numpy as np
import ctypes
import os
import pathlib

cimport numpy as cnp

cdef char* LANGUAGE = b'en'

cdef whisper_full_params default_params() nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = False
    params.print_progress = False 
    params.translate = False
    params.language = <const char *> LANGUAGE
    params.n_threads = 2
    params.speed_up = False
    params.print_timestamps = True
    params.suppress_blank = True
    params.suppress_non_speech_tokens = True
    params.temperature = 0.0
    params.max_initial_ts = 1.0
    params.length_penalty = -1.0
    params.temperature_inc = 0.2
    params.max_len = 30

    params.thold_pt = 0.01
    params.thold_ptsum = 0.01
    params.print_timestamps = False
    params.n_max_text_ctx = 16384

    params.entropy_thold =  2.40
    params.logprob_thold = -1.0
    params.no_speech_thold = 0.6

    params.greedy.best_of = 1
    params.beam_search.beam_size = -1
    params.beam_search.patience = -1.0

    return params

cdef class WhisperContextWrapper:
    cdef whisper_context * ctx
    flag: bool

    def __init__(
        self,
        model_name = None, 
        DEFAULT_MODEL_FILE_PREFIX='ggml-', 
        DEFAULT_MODEL_DIR = './whisper.cpp/models',
        absolute_file_path = None,
    ):
        assert model_name is not None or absolute_file_path is not None
        assert model_name is None or absolute_file_path is None

        self.flag = False
        self.set_context_from_file(
            model_name = model_name, 
            DEFAULT_MODEL_FILE_PREFIX = DEFAULT_MODEL_FILE_PREFIX, 
            DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR,
            absolute_file_path = absolute_file_path
        )
        assert self.flag == True

    def set_context_from_file(
        self,
        model_name = None,
        DEFAULT_MODEL_FILE_PREFIX='ggml-', 
        DEFAULT_MODEL_DIR = './whisper.cpp/models',
        absolute_file_path = None,
    ):
        assert model_name is not None or absolute_file_path is not None
        assert model_name is None or absolute_file_path is None

        if self.flag:
            self.destroy_current_context()

        assert self.flag == False
        model_path = pathlib.Path(DEFAULT_MODEL_DIR).joinpath(f"{DEFAULT_MODEL_FILE_PREFIX}{model_name}.bin") if absolute_file_path is None else absolute_file_path
        
        if not os.path.exists(model_path):
            print("your model path:")
            print(model_path)
            print()
            print("but NOT FOUND,\nRun following steps:\ncd ./whisper.cpp\nmake tiny\npython\n\n")
            print(">>> model = WhisperContextWrapper('tiny')")
            raise Exception("NOT FOUND model.")

        cdef bytes model_path_bytes = str(model_path).encode('utf8')
        self.ctx = whisper_init_from_file(model_path_bytes)

        self.flag = True

        whisper_print_system_info()
    
    def has_context(self):
        return self.flag

    def destroy_current_context(self):
        if self.flag:
            whisper_free(self.ctx)
        self.flag = False

    def __del__(self):
        self.destroy_current_context()

    def __dealloc__(self):
        self.destroy_current_context()

cdef class WhisperFullParamsWrapper:
    cdef whisper_full_params params
    def __init__(self):
        self.params = default_params() 

def transcribe(
    waveform,
    whisper_context_wrapper: WhisperContextWrapper,
    whisper_params_wrapper: WhisperFullParamsWrapper,
    temperature = 0.0, 
    max_len = 30,
    print_timestamps: bool = True,
    print_progress: bool = False,
    print_realtime: bool = False,
    no_speech_thold = 0.6,
    language = b'en',
    raise_exception_on_fail = False
):
    assert whisper_context_wrapper.has_context()

    cdef whisper_context * ctx = whisper_context_wrapper.ctx
    cdef whisper_full_params params = whisper_params_wrapper.params

    params.temperature = temperature
    params.print_timestamps = print_timestamps
    params.print_progress = print_progress
    params.language = language
    params.print_realtime = print_realtime
    params.no_speech_thold = no_speech_thold
    
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] data = waveform
        
    ret = whisper_full(ctx, params, &data[0], waveform.shape[0])
    if ret != 0 and raise_exception_on_fail:
        raise Exception("whisper execution fail")
    return ret

def extract_output(
    whisper_context_wrapper: WhisperContextWrapper, 
    with_token_text = False,
    with_token_id = False,
    with_token_data = False
):
    assert whisper_context_wrapper.has_context()

    cdef whisper_context * ctx = whisper_context_wrapper.ctx
    cdef int n_segments = whisper_full_n_segments(ctx)

    ret = []
    for i in range(n_segments):
        el = { 
            'text': whisper_full_get_segment_text(ctx, i).decode('utf-8', errors='replace'), 
            'start': whisper_full_get_segment_t0(ctx, i), 
            'end': whisper_full_get_segment_t1(ctx, i),
            'token_texts': [],
            'token_ids': [], 
            'token_datas': []
        }

        n_tokens = whisper_full_n_tokens(ctx, i)
        if with_token_text:
            el['token_texts'] = [whisper_full_get_token_text(ctx, i, t).decode('utf-8', errors='replace') for t in range(n_tokens)]
        if with_token_id: 
            el['token_ids'] = [whisper_full_get_token_id(ctx, i, t) for t in range(n_tokens)]
        if with_token_data:
            for t in range(n_tokens):
                token_data = whisper_full_get_token_data(ctx, i, t)
                el['token_datas'].append({'p': token_data.p, 'pt': token_data.pt, 'ptsum': token_data.ptsum, 'vlen': token_data.vlen})

        ret.append(el)

    return ret

