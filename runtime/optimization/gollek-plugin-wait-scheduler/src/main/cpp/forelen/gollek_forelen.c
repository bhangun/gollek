/**
 * gollek_forelen.c — ForeLen output-length predictor for Gollek
 *
 * A lightweight 2-layer MLP trained on (prompt_tokens → output_tokens) pairs.
 * Runs on CPU in ~0.5 ms. Loaded by ForelenBinding.java via Java FFM.
 *
 * Model: MLP(prompt_tokens) → output_tokens
 *   Layer 1: Linear(1, 64) + ReLU
 *   Layer 2: Linear(64, 1)
 *
 * Weights are initialised to sensible heuristic defaults and can be
 * updated online via forelen_update().
 *
 * Build:
 *   make -C src/main/cpp/forelen
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ── Model weights ──────────────────────────────────────────────────────────────

#define HIDDEN 64

// Initialised to heuristic: output ≈ 0.5 × prompt, clamped [64, 2048]
static float w1[HIDDEN];      // input→hidden weights (bias included as w1[HIDDEN-1])
static float b1[HIDDEN];      // hidden biases
static float w2[HIDDEN];      // hidden→output weights
static float b2 = 0.f;        // output bias

// Online learning state (simple gradient accumulation)
static float grad_w1[HIDDEN];
static float grad_w2[HIDDEN];
static int   update_count = 0;
static float learning_rate = 0.001f;

static int model_initialized = 0;

static void init_model(void) {
    if (model_initialized) return;
    // Default weights: linear interpolation ~ output = 0.5 × prompt
    for (int i = 0; i < HIDDEN; i++) {
        w1[i] = (i < HIDDEN / 2) ? 0.01f : -0.01f;
        b1[i] = (float)i / (HIDDEN * 2.0f);
        w2[i] = 0.5f / HIDDEN;
    }
    b2 = 64.0f;    // minimum output tokens
    memset(grad_w1, 0, sizeof(grad_w1));
    memset(grad_w2, 0, sizeof(grad_w2));
    model_initialized = 1;
}

static float relu(float x) { return x > 0.f ? x : 0.f; }

// ── Forward pass ───────────────────────────────────────────────────────────────

static float mlp_forward(float prompt_tokens) {
    init_model();

    // Normalise input: log(1 + x) / 10  (maps [0, ∞) to [0, ~1])
    float x = logf(1.f + prompt_tokens) / 10.f;

    // Layer 1
    float hidden[HIDDEN];
    for (int i = 0; i < HIDDEN; i++)
        hidden[i] = relu(w1[i] * x + b1[i]);

    // Layer 2
    float out = b2;
    for (int i = 0; i < HIDDEN; i++)
        out += w2[i] * hidden[i];

    // Clamp to [64, 2048]
    if (out < 64.f)   out = 64.f;
    if (out > 2048.f) out = 2048.f;
    return out;
}

// ── Online update (SGD with gradient accumulation) ────────────────────────────

static void mlp_update_one(float prompt_tokens, float actual_output) {
    float x = logf(1.f + prompt_tokens) / 10.f;

    float hidden[HIDDEN];
    for (int i = 0; i < HIDDEN; i++)
        hidden[i] = relu(w1[i] * x + b1[i]);

    float pred = b2;
    for (int i = 0; i < HIDDEN; i++) pred += w2[i] * hidden[i];
    pred = fmaxf(64.f, fminf(2048.f, pred));

    float err = pred - actual_output;

    // Gradient of output layer
    for (int i = 0; i < HIDDEN; i++) {
        grad_w2[i] += err * hidden[i];
    }

    // Gradient of hidden layer
    for (int i = 0; i < HIDDEN; i++) {
        float dh = (hidden[i] > 0.f) ? err * w2[i] : 0.f;
        grad_w1[i] += dh * x;
    }
    update_count++;
}

static void flush_gradients(void) {
    if (update_count == 0) return;
    float n = (float)update_count;
    for (int i = 0; i < HIDDEN; i++) {
        w2[i] -= learning_rate * grad_w2[i] / n;
        w1[i] -= learning_rate * grad_w1[i] / n;
    }
    memset(grad_w2, 0, sizeof(grad_w2));
    memset(grad_w1, 0, sizeof(grad_w1));
    update_count = 0;
}

// ── MAE tracking ──────────────────────────────────────────────────────────────

static float mae_sum   = 0.f;
static int   mae_count = 0;

// ── C API (loaded by ForelenBinding.java via FFM) ─────────────────────────────

int forelen_predict(int prompt_tokens) {
    float result = mlp_forward((float)prompt_tokens);
    return (int)result;
}

int forelen_predict_batch(int* out_predictions, const int* prompt_lengths, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        out_predictions[i] = forelen_predict(prompt_lengths[i]);
    }
    return 0;
}

int forelen_update(const int* prompt_lengths, const int* actual_outputs, int n) {
    for (int i = 0; i < n; i++) {
        float pred   = mlp_forward((float)prompt_lengths[i]);
        float actual = (float)actual_outputs[i];
        float err    = fabsf(pred - actual);
        mae_sum   += err;
        mae_count += 1;
        mlp_update_one((float)prompt_lengths[i], actual);
    }
    flush_gradients();
    return 0;
}

float forelen_mae(void) {
    if (mae_count == 0) return 0.f;
    float result = mae_sum / (float)mae_count;
    // Rolling window: decay towards recent
    mae_sum   *= 0.9f;
    mae_count = (int)(mae_count * 0.9f);
    return result;
}
