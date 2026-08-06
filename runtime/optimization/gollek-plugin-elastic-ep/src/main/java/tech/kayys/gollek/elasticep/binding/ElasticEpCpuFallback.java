package tech.kayys.gollek.elasticep.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for Elastic Expert Parallelism kernels.
 *
 * <p>Used for development and CI when {@code libgollek_ep.so} is absent.
 * EP dispatch is simulated as identity passthrough (no real expert routing);
 * rebalance uses a simple round-robin assignment.
 */
public final class ElasticEpCpuFallback {

    private static final Logger LOG = Logger.getLogger(ElasticEpCpuFallback.class);

    private ElasticEpCpuFallback() {}

    /**
     * CPU fallback: copies input to output (no real expert routing).
     * In production all routing happens on GPU — this is a correctness stub.
     */
    public static int epDispatch(
            MemorySegment out,
            MemorySegment input,
            MemorySegment expertIds,
            int B, int T, int topK, int numExperts, int dim) {

        LOG.debug("ElasticEP CPU fallback dispatch (native unavailable) — identity passthrough");
        long n = (long) B * T * dim;
        for (long i = 0; i < n; i++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    input.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        return 0;
    }

    /**
     * CPU fallback: round-robin expert → GPU assignment.
     */
    public static int epRebalance(
            MemorySegment newAssignment,
            MemorySegment loadHistogram,
            int numExperts, int numGpus) {

        LOG.debug("ElasticEP CPU fallback rebalance — round-robin assignment");
        for (int e = 0; e < numExperts; e++) {
            newAssignment.setAtIndex(ValueLayout.JAVA_INT, e, e % numGpus);
        }
        return 0;
    }
}
