package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;

final class OnnxTextWorkspacePool implements AutoCloseable {

    private final WorkspaceFactory factory;
    private final int maxEntries;
    private final Deque<OnnxTextRunWorkspace> idle;
    private boolean closed;

    private OnnxTextWorkspacePool(
            WorkspaceFactory factory,
            int maxEntries) {
        this.factory = Objects.requireNonNull(factory, "factory");
        if (maxEntries < 0) {
            throw new IllegalArgumentException("maxEntries must be >= 0");
        }
        this.maxEntries = maxEntries;
        this.idle = new ArrayDeque<>(Math.max(1, maxEntries));
    }

    static OnnxTextWorkspacePool create(
            OnnxRuntimeBinding binding,
            MemorySegment session,
            MemorySegment memInfo,
            OnnxTextSessionResources resources,
            int maxEntries,
            int inputTensorCacheEntries) {
        Objects.requireNonNull(binding, "binding");
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(memInfo, "memInfo");
        Objects.requireNonNull(resources, "resources");
        return new OnnxTextWorkspacePool(
                (promptLength, maxTokens) -> OnnxTextRunWorkspace.create(
                        binding,
                        session,
                        memInfo,
                        resources,
                        promptLength,
                        maxTokens,
                        inputTensorCacheEntries),
                maxEntries);
    }

    static OnnxTextWorkspacePool createForTest(WorkspaceFactory factory, int maxEntries) {
        return new OnnxTextWorkspacePool(factory, maxEntries);
    }

    Lease acquire(int promptLength, int maxTokens) {
        Acquisition acquisition = acquireWorkspace(promptLength, maxTokens);
        acquisition.workspace().beginRequest();
        return new Lease(
                this,
                acquisition.workspace(),
                acquisition.reused(),
                acquisition.evicted(),
                OnnxTextRunWorkspace.capacityFor(promptLength, maxTokens),
                acquisition.workspace().capacity());
    }

    synchronized int idleCount() {
        return idle.size();
    }

    private synchronized Acquisition acquireWorkspace(int promptLength, int maxTokens) {
        ensureOpen();
        OnnxTextRunWorkspace best = null;
        int bestCapacity = Integer.MAX_VALUE;
        for (OnnxTextRunWorkspace candidate : idle) {
            if (candidate.canServe(promptLength, maxTokens) && candidate.capacity() < bestCapacity) {
                best = candidate;
                bestCapacity = candidate.capacity();
            }
        }
        if (best != null) {
            idle.remove(best);
            return new Acquisition(best, true, 0);
        }
        return new Acquisition(factory.create(promptLength, maxTokens), false, 0);
    }

    interface WorkspaceFactory {
        OnnxTextRunWorkspace create(int promptLength, int maxTokens);
    }

    private record Acquisition(OnnxTextRunWorkspace workspace, boolean reused, int evicted) {
    }

    private synchronized int release(OnnxTextRunWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.finishRequest();
        if (closed || maxEntries == 0) {
            workspace.close();
            return 0;
        }
        if (idle.size() < maxEntries) {
            idle.addFirst(workspace);
            return 0;
        }
        OnnxTextRunWorkspace smallest = smallestIdleWorkspace();
        if (smallest != null && workspace.capacity() > smallest.capacity()) {
            idle.remove(smallest);
            smallest.close();
            idle.addFirst(workspace);
            return 1;
        }
        workspace.close();
        return 1;
    }

    private OnnxTextRunWorkspace smallestIdleWorkspace() {
        OnnxTextRunWorkspace smallest = null;
        int smallestCapacity = Integer.MAX_VALUE;
        for (OnnxTextRunWorkspace candidate : idle) {
            if (candidate.capacity() < smallestCapacity) {
                smallest = candidate;
                smallestCapacity = candidate.capacity();
            }
        }
        return smallest;
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        while (!idle.isEmpty()) {
            idle.removeFirst().close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ONNX text workspace pool is already closed");
        }
    }

    static final class Lease implements AutoCloseable {
        private final OnnxTextWorkspacePool owner;
        private final boolean reused;
        private final int requestedCapacity;
        private final int workspaceCapacity;
        private int evicted;
        private OnnxTextRunWorkspace workspace;

        private Lease(
                OnnxTextWorkspacePool owner,
                OnnxTextRunWorkspace workspace,
                boolean reused,
                int evicted,
                int requestedCapacity,
                int workspaceCapacity) {
            this.owner = Objects.requireNonNull(owner, "owner");
            this.workspace = Objects.requireNonNull(workspace, "workspace");
            this.reused = reused;
            this.evicted = evicted;
            this.requestedCapacity = requestedCapacity;
            this.workspaceCapacity = workspaceCapacity;
        }

        OnnxTextRunWorkspace workspace() {
            if (workspace == null) {
                throw new IllegalStateException("ONNX text workspace lease is already closed");
            }
            return workspace;
        }

        boolean reused() {
            return reused;
        }

        int evicted() {
            return evicted;
        }

        int requestedCapacity() {
            return requestedCapacity;
        }

        int workspaceCapacity() {
            return workspaceCapacity;
        }

        @Override
        public void close() {
            OnnxTextRunWorkspace current = workspace;
            if (current == null) {
                return;
            }
            workspace = null;
            evicted += owner.release(current);
        }
    }
}
