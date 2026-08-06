/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Regression coverage for wrapper-prefix tensor aliases applied after SafeTensor loading.
 */
class WeightAliasExpanderTest {

    @Test
    void commonAliasesExposeWrappedLmHeadAtCanonicalRoot() {
        AccelTensor lmHead = AccelTensor.zeros(4, 3);
        Map<String, AccelTensor> weights = new HashMap<>();
        weights.put("model.language_model.lm_head.weight", lmHead);

        WeightAliasExpander.applyCommonAliases(weights);

        assertSame(lmHead, weights.get("lm_head.weight"));
        assertSame(lmHead, weights.get("model.lm_head.weight"));
    }

    @Test
    void commonAliasesExposeNestedTextWrapperLmHeadAtCanonicalRoot() {
        AccelTensor textModelHead = AccelTensor.zeros(4, 3);
        AccelTensor languageModelBias = AccelTensor.zeros(4);
        Map<String, AccelTensor> weights = new HashMap<>();
        weights.put("text_model.model.lm_head.weight", textModelHead);
        weights.put("language_model.model.lm_head.bias", languageModelBias);

        WeightAliasExpander.applyCommonAliases(weights);

        assertSame(textModelHead, weights.get("lm_head.weight"));
        assertSame(textModelHead, weights.get("model.lm_head.weight"));
        assertSame(languageModelBias, weights.get("lm_head.bias"));
        assertSame(languageModelBias, weights.get("model.lm_head.bias"));
    }

    @Test
    void commonAliasesDoNotOverwriteExistingCanonicalLmHead() {
        AccelTensor canonical = AccelTensor.zeros(4, 3);
        AccelTensor wrapped = AccelTensor.zeros(4, 3);
        Map<String, AccelTensor> weights = new HashMap<>();
        weights.put("lm_head.weight", canonical);
        weights.put("model.language_model.lm_head.weight", wrapped);

        WeightAliasExpander.applyCommonAliases(weights);

        assertSame(canonical, weights.get("lm_head.weight"));
        assertSame(wrapped, weights.get("model.lm_head.weight"));
    }
}
