package tech.kayys.gollek.runtime.batch;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

public final class ContinuousBatchScheduler {
    private final Queue<DecodeRequest> queue = new ConcurrentLinkedQueue<>();

    public void submit(DecodeRequest req) {
        queue.offer(req);
    }

    public List<DecodeRequest> nextBatch(int maxBatch) {
        List<DecodeRequest> batch = new ArrayList<>();
        for (int i = 0; i < maxBatch; i++) {
            DecodeRequest r = queue.poll();
            if (r == null)
                break;
            if (!r.finished) {
                batch.add(r);
            }
        }
        return batch;
    }

    public void requeue(List<DecodeRequest> reqs) {
        for (DecodeRequest r : reqs) {
            if (!r.finished)
                queue.offer(r);
        }
    }
}