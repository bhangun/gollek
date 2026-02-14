#include "llama.h"

static void golek_llama_noop_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

void golek_llama_log_disable(void) {
    llama_log_set(golek_llama_noop_log_callback, NULL);
}

void golek_llama_model_default_params_into(struct llama_model_params *out) {
    if (!out) {
        return;
    }
    *out = llama_model_default_params();
}

void golek_llama_context_default_params_into(struct llama_context_params *out) {
    if (!out) {
        return;
    }
    *out = llama_context_default_params();
}
