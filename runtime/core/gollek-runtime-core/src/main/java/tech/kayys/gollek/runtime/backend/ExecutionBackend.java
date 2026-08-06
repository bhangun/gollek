package tech.kayys.gollek.runtime.backend;

import tech.kayys.gollek.runtime.plan.ExecutionPlan;
import tech.kayys.gollek.runtime.control.ExecutionSession;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.Map;

/**
 * 
 * IR → Compiler → ExecutionPlan
 * ↓
 * ExecutionEngine (single-run, local)
 * ↓
 * Runtime Orchestrator ← 🔥 NEW LAYER
 * ↓
 * ┌────────────────┼────────────────┐
 * │ │ │
 * Local Exec Remote Exec Hybrid Exec
 * 
 * ExecutionPlan → what to run
 * ExecutionEngine → how to run (single node)
 * ExecutionOrchestrator → where & when to run
 */
public interface ExecutionBackend {
    Map<String, Tensor> execute(
            ExecutionPlan plan,
            Map<String, Tensor> inputs,
            ExecutionSession session);
}