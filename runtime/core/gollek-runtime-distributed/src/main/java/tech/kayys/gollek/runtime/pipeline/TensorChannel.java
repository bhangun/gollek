package tech.kayys.gollek.runtime.pipeline;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.concurrent.*;

public final class TensorChannel {
    private final BlockingQueue<Tensor> queue = new LinkedBlockingQueue<>();

    public void send(Tensor t) {
        queue.offer(t);
    }

    public Tensor recv() throws InterruptedException {
        return queue.take();
    }
}
