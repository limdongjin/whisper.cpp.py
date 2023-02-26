from ctypes import Structure, c_int, c_float, c_bool, c_void_p, c_int64, c_int32, c_uint, c_char_p

class WhisperHparams(Structure):
    _fields_ = [
        ("n_vocab", c_int32),
        ("n_audio_ctx", c_int32),
        ("n_audio_state", c_int32),
        ("n_audio_head", c_int32),
        ("n_audio_layer", c_int32),
        ("n_text_ctx", c_int32),
        ("n_text_state", c_int32),
        ("n_text_head", c_int32),
        ("n_text_layer", c_int32),
        ("n_mels", c_int32),
        ("f16", c_int32)
    ]

class WhisperFilters(Structure):
    _fields_ = [
        ("n_mel", c_int32),
        ("n_fft", c_int32),
        ("data", c_void_p)
    ]

class WhisperMel(Structure):
    _fields_ = [
        ("n_len", c_int),
        ("n_mel", c_int),
        ("data", c_void_p)
    ]

class WhisperModel(Structure):
    _fields_ = [
        ("type", c_uint),
#
        ("hparams", WhisperHparams),
        ("filters", WhisperFilters),
#       
        ("e_pe", c_void_p),
#
        ("e_conv_1_w", c_void_p),
        ("e_conv_1_b", c_void_p),
#
        ("e_conv_2_w", c_void_p),
        ("e_conv_2_b", c_void_p),
#
        ("e_ln_w", c_void_p),
        ("e_ln_b", c_void_p),
#
        ("d_pe", c_void_p),
        ("d_te", c_void_p),
#
        ("d_ln_w", c_void_p),
        ("d_ln_b", c_void_p),
#
        ("layers_encoder", c_void_p),
        ("layers_decoder", c_void_p),
#
        ("ctx", c_void_p), # ggml_context*
#
        ("buf", c_void_p),
#
        ("n_loaded", c_int),
        ("tensors", c_void_p)
    ]


class WhisperContext(Structure):
    _fields_ = [
        ("t_load_us", c_int64),
        ("t_mel_us", c_int64),
        ("t_sample_us", c_int64),
        ("t_encode_us", c_int64),
        ("t_decode_us", c_int64),
        ("t_start_us", c_int64),
#
        ("n_sample", c_int32),
        ("n_encode", c_int32),
        ("n_decode", c_int32),
        ("n_fall_p", c_int32),
        ("n_fall_h", c_int32),
#
        ("wtype", c_uint),
        ("mel", WhisperMel),
        ("model", WhisperModel),
        ("vocab", c_void_p),
        ("kv_cross", c_void_p),
        ("decoders", c_void_p),
        ("buf_compute", c_void_p),
        ("buf_scratch", c_void_p),
        ("buf_last", c_int),
        ("buf_max_size", c_int64),
        ("logits", c_void_p),
        ("result_all", c_void_p),
        ("prompt_past", c_void_p),
        ("logits_id", c_void_p),
        ("rng", c_void_p),
        ("lang_id", c_int),
        ("t_beg", c_int64),
        ("t_last", c_int64),
        ("tid_last", c_int),
        ("energy", c_void_p),
        ("exp_n_audio_ctx", c_int32)
    ]

class GreedyParam(Structure):
    _fields_ = [
        ("best_of", c_int)
    ]

class BeamSearchParam(Structure):
    _fields_ = [
        ("beam_size", c_int),
        ("patience", c_float)
    ]

class WhisperFullParams(Structure):
    _fields_ = [
        ("strategy", c_int), 
#
        ("n_threads", c_int),
        ("n_max_text_ctx", c_int),
        ("offset_ms", c_int),
        ("duration_ms", c_int),
#
        ("translate", c_bool),
        ("no_context", c_bool),
        ("single_segment", c_bool),
        ("print_special", c_bool),
        ("print_progress", c_bool),
        ("print_realtime", c_bool),
        ("print_timestamps", c_bool),
#
        ("token_timestamps", c_bool),
        ("thold_pt", c_float),
        ("thold_ptsum", c_float),
        ("max_len", c_int),
        ("split_on_word", c_bool),
        ("max_tokens", c_int),
#
        ("speed_up", c_bool),
        ("audio_ctx", c_int),
#
        ("prompt_tokens", c_void_p),
        ("prompt_n_tokens", c_int),
#
        ("language", c_char_p),
#
        ("suppress_blank", c_bool),
        ("suppress_non_speech_tokens", c_bool),
#
        ("temperature", c_float),
        ("max_initial_ts", c_float),
        ("length_penalty", c_float),
#
        ("temperature_inc", c_float),
        ("entropy_thold", c_float),
        ("logprob_thold", c_float),
        ("no_speech_thold", c_float),
#
        ("greedy", GreedyParam),
#
        ("beam_search", BeamSearchParam),
#
        ("new_segment_callback", c_void_p),
        ("new_segment_callback_user_data", c_void_p),
        ("encoder_begin_callback", c_void_p),
        ("encoder_begin_callback_user_data", c_void_p),
        ("logits_filter_callback", c_void_p),
        ("logits_filter_callback_user_data", c_void_p),
    ]

class WhisperTokenData(Structure):
    _fields_ = [
        ('id', c_int),
        ('tid', c_int),
        ('p', c_float),
        ('plog', c_float),
        ('pt', c_float),
        ('ptsum', c_float),
        ('t0', c_int64),
        ('t1', c_int64),
        ('vlen', c_float)
    ]

