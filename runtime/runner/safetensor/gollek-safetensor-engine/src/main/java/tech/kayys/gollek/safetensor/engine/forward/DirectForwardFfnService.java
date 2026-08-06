/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

/**
 * Public FFN entry point for callers that need direct feed-forward execution
 * without depending on the full transformer forward facade.
 */
@ApplicationScoped
public class DirectForwardFfnService {
    private static final Logger log = Logger.getLogger(DirectForwardFfnService.class);

    @Inject
    DirectForwardRuntimeState runtimeState;
    @Inject
    DirectForwardModelContext modelContext;

    AccelTensor nonGated(AccelTensor x, ModelConfig config, AccelTensor upW, AccelTensor downW) {
        DirectForwardOperators operators = operators();
        AccelTensor up = operators.linear(x, upW, null, "ffn_up_nongated", config);
        AccelTensor act = AccelOps.silu(up);
        up.close();
        AccelTensor out = operators.ffnDownLinear(act, downW, null, config, "ffn_down_nongated");
        act.close();
        return out;
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW,
            AccelTensor gateB, AccelTensor upW, AccelTensor upB, AccelTensor downW, AccelTensor downB) {
        return swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW,
            AccelTensor gateB, AccelTensor upW, AccelTensor upB, AccelTensor downW, AccelTensor downB,
            ForwardWorkspace ws) {
        return swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, ws, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW,
            AccelTensor gateB, AccelTensor upW, AccelTensor upB, AccelTensor downW, AccelTensor downB,
            ForwardWorkspace ws, AccelTensor downOutputBuffer) {
        return swigluFfn(x, arch, config,
                new DirectForwardGatedFfnWeights(gateW, gateB, upW, upB, downW, downB), ws, downOutputBuffer);
    }

    AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config,
            DirectForwardGatedFfnWeights weights, ForwardWorkspace ws, AccelTensor downOutputBuffer) {
        return operators().swigluFfn(x, arch, config, weights, ws, downOutputBuffer);
    }

    private DirectForwardOperators operators() {
        return new DirectForwardOperators(runtimeState.context(log), modelContext::traits);
    }
}
