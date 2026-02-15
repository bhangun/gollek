package io.github.pytorch.optim;

import io.github.pytorch.core.Tensor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Base class for all optimizers
 */
public abstract class Optimizer implements AutoCloseable {
    
    protected final List<Tensor> parameters;
    protected final Arena arena;
    protected MemorySegment nativeHandle;
    protected boolean closed = false;
    
    public Optimizer(List<Tensor> parameters) {
        this.parameters = parameters;
        this.arena = Arena.ofConfined();
    }
    
    /**
     * Perform a single optimization step
     */
    public abstract void step();
    
    /**
     * Zero all gradients
     */
    public void zeroGrad() {
        for (Tensor param : parameters) {
            param.zeroGrad();
        }
    }
    
    /**
     * Get parameters
     */
    public List<Tensor> getParameters() {
        return parameters;
    }
    
    protected void checkClosed() {
        if (closed) {
            throw new IllegalStateException("Optimizer has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            if (arena != null) {
                arena.close();
            }
            closed = true;
        }
    }
}
