#!python
# cython: language_level=3
from libc.stdint cimport int64_t

cdef nogil:
    int WHISPER_SAMPLE_RATE = 16000
    int WHISPER_N_FFT = 400
    int WHISPER_N_MEL = 80
    int WHISPER_HOP_LENGTH = 160
    int WHISPER_CHUNK_SIZE = 30
    char* LANGUAGE = b'en'
    ctypedef struct audio_data:
        float* frames;
        int n_frames;

cdef extern from "whisper.h" nogil:
    enum whisper_sampling_strategy:
        WHISPER_SAMPLING_GREEDY = 0,
        WHISPER_SAMPLING_BEAM_SEARCH,
    ctypedef bint _Bool
    ctypedef void (*whisper_new_segment_callback)(whisper_context*, int, void*)
    ctypedef _Bool whisper_encoder_begin_callback(whisper_context*, void*)
    ctypedef int whisper_token
    ctypedef struct whisper_token_data:
        whisper_token id
        whisper_token tid
        float p
        float plog
        float pt
        float ptsum
        int64_t t0
        int64_t t1
        float vlen
    ctypedef void (*whisper_logits_filter_callback)(whisper_context*, whisper_token_data*, int, float*, void*)
    ctypedef struct whisper_context:
        pass
    ctypedef struct grd:
        int best_of
    ctypedef struct beam:
        int beam_size
        float patience
    ctypedef struct whisper_model_loader:
        void* context
        size_t (*read)(void * ctx, void * output, size_t read_size)
        _Bool (*eof)(void * ctx)
        void  (*close)(void * ctx)
    cdef whisper_context* whisper_init_from_file(char*)
    cdef whisper_context* whisper_init_from_buffer(void*, size_t)
    cdef whisper_context* whisper_init(whisper_model_loader*)
    cdef int whisper_pcm_to_mel_phase_vocoder(whisper_context*, float*, int, int)
    ctypedef struct whisper_full_params:
        int strategy
        int n_threads
        int n_max_text_ctx
        int offset_ms
        int duration_ms
        _Bool translate
        _Bool no_context
        _Bool single_segment
        _Bool print_special
        _Bool print_progress
        _Bool print_realtime
        _Bool print_timestamps
        _Bool token_timestamps
        float thold_pt
        float thold_ptsum
        int max_len
        _Bool split_on_word
        int max_tokens
        _Bool speed_up
        int audio_ctx
        whisper_token* prompt_tokens
        int prompt_n_tokens
        char* language
        _Bool suppress_blank
        _Bool suppress_non_speech_tokens
        float temperature
        float max_initial_ts
        float length_penalty
        float temperature_inc
        float entropy_thold
        float logprob_thold
        float no_speech_thold
        grd greedy
        beam beam_search
        whisper_new_segment_callback new_segment_callback
        void* new_segment_callback_user_data
        whisper_encoder_begin_callback encoder_begin_callback
        void* encoder_begin_callback_user_data
        whisper_logits_filter_callback logits_filter_callback
        void * logits_filter_callback_user_data
    whisper_full_params whisper_full_default_params(whisper_sampling_strategy)
    cdef void whisper_free(whisper_context*)
    cdef int whisper_pcm_to_mel(whisper_context*, float*, int, int)
    cdef int whisper_set_mel(whisper_context*, float*, int, int)
    cdef int whisper_encode(whisper_context*, int, int)
    cdef int whisper_decode(whisper_context*, whisper_token*, int, int, int)
    cdef int whisper_n_audio_ctx(whisper_context*)
    cdef float* whisper_get_logits(whisper_context*)
    cdef int whisper_lang_id(char*)
    cdef int whisper_n_len(whisper_context*)
    cdef int whisper_n_vocab(whisper_context*)
    cdef int whisper_n_text_ctx(whisper_context*)
    cdef int whisper_is_multilingual(whisper_context*)
    cdef float* whisper_get_probs(whisper_context*)
    cdef whisper_token whisper_token_eot(whisper_context*)
    cdef whisper_token whisper_token_sot(whisper_context*)
    cdef whisper_token whisper_token_prev(whisper_context*)
    cdef whisper_token whisper_token_solm(whisper_context*)
    cdef whisper_token whisper_token_not(whisper_context*)
    cdef whisper_token whisper_token_beg(whisper_context*)
    cdef whisper_token whisper_token_translate()
    cdef whisper_token whisper_token_transcribe()
    cdef void whisper_print_timings(whisper_context*)
    cdef void whisper_reset_timings(whisper_context*)
    cdef int whisper_full(whisper_context*, whisper_full_params, float*, int)
    cdef int whisper_full_parallel(whisper_context*, whisper_full_params, float*, int, int)
    cdef int whisper_full_n_segments(whisper_context*)
    cdef int whisper_full_lang_id(whisper_context*)
    cdef int64_t whisper_full_get_segment_t0(whisper_context*, int)
    cdef int64_t whisper_full_get_segment_t1(whisper_context*, int)
    cdef int whisper_full_n_tokens(whisper_context*, int)
    cdef whisper_token whisper_full_get_token_id(whisper_context*, int, int)
    cdef char* whisper_full_get_token_text(whisper_context*, int, int)
    cdef whisper_token_data whisper_full_get_token_data(whisper_context*, int, int)
    cdef float whisper_full_get_token_p(whisper_context*, int, int)
    cdef int whisper_bench_memcpy(int)
    cdef int whisper_bench_ggml_mul_mat(int)
    const char* whisper_print_system_info()
    const char* whisper_full_get_segment_text(whisper_context*, int)
    

