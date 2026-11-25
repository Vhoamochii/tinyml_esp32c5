/*
 * Case 3: lop cuoi tuyen tinh C lop, softmax + cross-entropy, SGD mot buoc FP32.
 * Scalar -O3, KHONG SIMD. Hanh vi khop LinearHead<C,F> cua Rust.
 * Quy uoc: w la mang [C][F] row-major (w[c*F+i]); z da giai luong tu boi caller.
 * Error case: label >= C -> tra NaN, khong cap nhat.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>

void sgd_scores(const float* w, const float* b, const float* z,
                float* scores_out, int C, int F) {
    for (int c = 0; c < C; c++) {
        float s = b[c];
        const float* wrow = w + c * F;
        for (int i = 0; i < F; i++) {
            s += wrow[i] * z[i];
        }
        scores_out[c] = s;
    }
}

void sgd_softmax(const float* scores_in, float* p_out, int C) {
    /* Softmax on dinh: tru max de khong tran exp */
    float m = scores_in[0];
    for (int c = 1; c < C; c++) {
        if (scores_in[c] > m) m = scores_in[c];
    }
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        float e = expf(scores_in[c] - m);
        p_out[c] = e;
        sum += e;
    }
    for (int c = 0; c < C; c++) p_out[c] /= sum;
}

uint8_t sgd_predict(const float* w, const float* b, const float* z, int C, int F) {
    float best_s = -FLT_MAX;
    uint8_t best = 0;
    for (int c = 0; c < C; c++) {
        float s = b[c];
        const float* wrow = w + c * F;
        for (int i = 0; i < F; i++) s += wrow[i] * z[i];
        if (s > best_s) { best_s = s; best = (uint8_t)c; }
    }
    return best;
}

/* Mot buoc SGD. Tra ve loss cross-entropy. label >= C -> NaN, khong cap nhat. */
float sgd_step(float* w, float* b, const float* z,
               uint8_t label, float lr, int C, int F) {
    if ((int)label >= C) {
        return (float)NAN;
    }
    float scores[8]; /* du cho C <= 8 */
    if (C > 8) return (float)NAN;
    for (int c = 0; c < C; c++) {
        float s = b[c];
        const float* wrow = w + c * F;
        for (int i = 0; i < F; i++) s += wrow[i] * z[i];
        scores[c] = s;
    }
    float p[8];
    {
        float m = scores[0];
        for (int c = 1; c < C; c++) if (scores[c] > m) m = scores[c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) { p[c] = expf(scores[c] - m); sum += p[c]; }
        for (int c = 0; c < C; c++) p[c] /= sum;
    }
    /* Khong cong epsilon: khop tuyet doi voi -lnf(p[y]) ben Rust */
    float loss = -logf(p[label]);
    for (int c = 0; c < C; c++) {
        float g = p[c] - (c == (int)label ? 1.0f : 0.0f);
        b[c] -= lr * g;
        float* wrow = w + c * F;
        for (int i = 0; i < F; i++) {
            wrow[i] -= lr * g * z[i];
        }
    }
    return loss;
}
