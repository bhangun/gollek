/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

final class TokenSamplingCandidates {
    private TokenSamplingCandidates() {
    }

    static Prepared prepare(float[] logits, int topK, int[] indices) {
        for (int i = 0; i < logits.length; i++) {
            indices[i] = i;
        }

        int limit = candidateLimit(logits.length, topK);
        if (limit < logits.length) {
            quickSelect(logits, indices, 0, logits.length - 1, limit);
        }

        iterativeQuickSort(logits, indices, 0, limit - 1, limit);
        return new Prepared(indices, limit);
    }

    private static int candidateLimit(int logitsLength, int topK) {
        return topK > 0 && topK < logitsLength ? topK : logitsLength;
    }

    private static void iterativeQuickSort(float[] values, int[] indices, int low, int high, int limit) {
        if (low >= high) {
            return;
        }

        int[] stack = new int[(high - low + 1) * 2];
        int top = -1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];

            int p = partition(values, indices, low, high);

            if (low < p - 1) {
                stack[++top] = low;
                stack[++top] = p - 1;
            }

            if (p + 1 < high && p < limit) {
                stack[++top] = p + 1;
                stack[++top] = high;
            }
        }
    }

    private static void quickSelect(float[] values, int[] indices, int low, int high, int k) {
        while (low < high) {
            int p = partition(values, indices, low, high);
            if (p == k) {
                return;
            }
            if (k < p) {
                high = p - 1;
            } else {
                low = p + 1;
            }
        }
    }

    private static int partition(float[] values, int[] indices, int low, int high) {
        float pivot = values[indices[high]];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++) {
            if (values[indices[j]] > pivot) {
                i++;
                swap(indices, i, j);
            }
        }
        swap(indices, i + 1, high);
        return i + 1;
    }

    private static void swap(int[] indices, int left, int right) {
        int temp = indices[left];
        indices[left] = indices[right];
        indices[right] = temp;
    }

    record Prepared(int[] indices, int limit) {
    }
}
