package tech.kayys.gollek.runtime.batch;

import tech.kayys.gollek.runtime.plan.ExecutionPlan;
import tech.kayys.gollek.runtime.kv.*;
import tech.kayys.gollek.runtime.ExecutionEngine;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.*;

public final class BatchDecodeEngine {
    private final ExecutionEngine engine;
    private final PagedKVCache kvPool;

    public BatchDecodeEngine(ExecutionEngine engine,
            PagedKVCache kvPool) {
        this.engine = engine;
        this.kvPool = kvPool;
    }

    public void step(List<DecodeRequest> batch,
            ExecutionPlan plan) {
        Map<String, Tensor> mergedInputs = new HashMap<>();
        // 🔥 merge inputs (simplified)
        for (DecodeRequest r : batch) {
            mergedInputs.put("input_" + r.id, buildInput(r));
        }
        Map<String, Tensor> outputs = engine.run(plan, mergedInputs, null);
        for (DecodeRequest r : batch) {
            int token = sample(outputs, r.id);
            r.streamer.onToken(token);
            if (isEnd(token)) {
                r.finished = true;
                r.streamer.onComplete();
                continue;
            }
            KVPage page = kvPool.acquire();
            // store KV (simplified)
            page.key = outputs.get("k_" + r.id);
            page.value = outputs.get("v_" + r.id);
            r.position++;
        }
    }

    private Tensor buildInput(DecodeRequest r) {
        return null; // integrate real tensor
    }

    private int sample(Map<String, Tensor> out, String id) {
        return 0; // sampling logic
    }

    private boolean isEnd(int token) {
        return token == -1;
    }
}