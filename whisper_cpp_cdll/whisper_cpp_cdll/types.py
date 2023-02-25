import ctypes

class GreedyParam(ctypes.Structure):
    _fields_ = [
        ("best_of", ctypes.c_int)
    ]

class BeamSearchParam(ctypes.Structure):
    _fields_ = [
        ("beam_size", ctypes.c_int),
        ("patience", ctypes.c_float)
    ]

class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int), 
#
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
#
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
#
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
#
        ("speed_up", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
#
        ("prompt_tokens", ctypes.c_void_p),
        ("prompt_n_tokens", ctypes.c_int),
#
        ("language", ctypes.c_char_p),
#
        ("suppress_blank", ctypes.c_bool),
        ("suppress_non_speech_tokens", ctypes.c_bool),
#
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
#
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
#
#        ("greedy", ctypes.c_int * 1),
        ("greedy", GreedyParam),
#
#        ("beam_search", ctypes.c_int * 2),
        ("beam_search", BeamSearchParam),
#
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
    ]

class WhisperTokenData(ctypes.Structure):
    _fields_ = [
        ('id', ctypes.c_int),
        ('tid', ctypes.c_int),
        ('p', ctypes.c_float),
        ('plog', ctypes.c_float),
        ('pt', ctypes.c_float),
        ('ptsum', ctypes.c_float),
        ('t0', ctypes.c_int64),
        ('t1', ctypes.c_int64),
        ('vlen', ctypes.c_float)
    ]

