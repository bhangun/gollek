package tech.kayys.gollek.ml.distributed;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Distributed Data Parallel (DDP) training across multiple nodes using TCP sockets.
 *
 * <p>Implements the ring-allreduce algorithm for gradient synchronization:
 * each worker sends its gradients to the next worker in a ring, accumulating
 * partial sums until all workers have the global average.
 *
 * <p>Uses JDK 25 virtual threads for non-blocking socket I/O.
 *
 * <h3>Architecture</h3>
 * <pre>
 *   Worker 0 ──→ Worker 1 ──→ Worker 2 ──→ Worker 0  (ring)
 *   Each step: scatter-reduce then all-gather
 * </pre>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // On each node:
 * var ddp = DistributedDataParallel.builder()
 *     .model(model)
 *     .rank(0)
 *     .worldSize(4)
 *     .masterAddr("192.168.1.1")
 *     .masterPort(29500)
 *     .build();
 *
 * ddp.barrier();          // sync before training
 * loss.backward();
 * ddp.allReduceGradients(); // average grads across all nodes
 * optimizer.step();
 * ddp.close();
 * }</pre>
 */
public final class DistributedDataParallel implements AutoCloseable {

    /** Rank of this worker (0-indexed). */
    private final int rank;

    /** Total number of workers. */
    private final int worldSize;

    /** The model whose gradients are synchronized. */
    private final NNModule model;

    /** TCP server socket accepting connections from the previous worker in the ring. */
    private final ServerSocket serverSocket;

    /** TCP connection to the next worker in the ring. */
    private Socket nextSocket;
    private DataOutputStream nextOut;
    private DataInputStream  prevIn;

    private DistributedDataParallel(Builder b) throws IOException {
        this.rank      = b.rank;
        this.worldSize = b.worldSize;
        this.model     = b.model;

        // Listen for the previous worker's connection
        this.serverSocket = new ServerSocket(b.masterPort + rank);

        // Connect to next worker in ring (rank+1) % worldSize
        int nextRank = (rank + 1) % worldSize;
        String nextAddr = rank == worldSize - 1 ? b.masterAddr : "localhost";
        int nextPort = b.masterPort + nextRank;

        // Use virtual thread to connect asynchronously while server waits
        Future<Socket> connectFuture;
        try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
            connectFuture = exec.submit(() -> {
                // Retry until next worker is ready
                for (int attempt = 0; attempt < 30; attempt++) {
                    try { return new Socket(nextAddr, nextPort); }
                    catch (IOException e) { Thread.sleep(200); }
                }
                throw new IOException("Could not connect to next worker at " + nextAddr + ":" + nextPort);
            });
            Socket prevSocket = serverSocket.accept(); // accept from previous worker
            this.nextSocket = connectFuture.get(10, TimeUnit.SECONDS);
            this.nextOut = new DataOutputStream(new BufferedOutputStream(nextSocket.getOutputStream()));
            this.prevIn  = new DataInputStream(new BufferedInputStream(prevSocket.getInputStream()));
        } catch (Exception e) {
            throw new IOException("DDP init failed for rank " + rank, e);
        }
    }

    /**
     * Performs a ring-allreduce to average gradients across all workers.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Scatter-reduce: each worker sends its gradient chunk to the next,
     *       accumulating partial sums over {@code worldSize-1} steps</li>
     *   <li>All-gather: each worker broadcasts its reduced chunk to all others</li>
     *   <li>Scale by {@code 1/worldSize} using {@link VectorOps#mulScalar}</li>
     * </ol>
     *
     * @throws IOException if network communication fails
     */
    public void allReduceGradients() throws IOException {
        List<Parameter> params = model.parameters();

        // Flatten all gradients into one buffer
        int totalSize = params.stream()
            .filter(p -> p.data().grad() != null)
            .mapToInt(p -> (int) p.data().grad().numel())
            .sum();

        float[] flatGrad = new float[totalSize];
        int offset = 0;
        for (Parameter p : params) {
            if (p.data().grad() == null) continue;
            float[] g = p.data().grad().data();
            System.arraycopy(g, 0, flatGrad, offset, g.length);
            offset += g.length;
        }

        // Ring-allreduce
        float[] reduced = ringAllReduce(flatGrad);

        // Scale by 1/worldSize using VectorOps (SIMD)
        VectorOps.mulScalar(reduced, 1.0f / worldSize, reduced);

        // Write back to parameter gradients
        offset = 0;
        for (Parameter p : params) {
            if (p.data().grad() == null) continue;
            float[] g = p.data().grad().data();
            System.arraycopy(reduced, offset, g, 0, g.length);
            offset += g.length;
        }
    }

    /**
     * Synchronizes all workers at a barrier — blocks until all workers arrive.
     *
     * @throws IOException if communication fails
     */
    public void barrier() throws IOException {
        // Send a single byte to next, receive from prev — repeat worldSize-1 times
        for (int i = 0; i < worldSize - 1; i++) {
            nextOut.writeByte(1);
            nextOut.flush();
            prevIn.readByte();
        }
    }

    // ── Ring-allreduce ────────────────────────────────────────────────────

    /**
     * Performs ring-allreduce on a flat gradient buffer.
     * Scatter-reduce phase followed by all-gather phase.
     */
    private float[] ringAllReduce(float[] data) throws IOException {
        int n = data.length;
        int chunkSize = (n + worldSize - 1) / worldSize;
        float[] buf = data.clone();

        // Scatter-reduce: worldSize-1 steps
        for (int step = 0; step < worldSize - 1; step++) {
            int sendChunk = ((rank - step + worldSize) % worldSize);
            int recvChunk = ((rank - step - 1 + worldSize) % worldSize);
            int sendOff = sendChunk * chunkSize;
            int recvOff = recvChunk * chunkSize;
            int sendLen = Math.min(chunkSize, n - sendOff);
            int recvLen = Math.min(chunkSize, n - recvOff);
            if (sendLen <= 0 || recvLen <= 0) continue;

            sendFloats(buf, sendOff, sendLen);
            float[] recv = receiveFloats(recvLen);
            for (int i = 0; i < recvLen; i++) buf[recvOff + i] += recv[i];
        }

        // All-gather: worldSize-1 steps
        for (int step = 0; step < worldSize - 1; step++) {
            int sendChunk = ((rank - step + 1 + worldSize) % worldSize);
            int recvChunk = ((rank - step + worldSize) % worldSize);
            int sendOff = sendChunk * chunkSize;
            int recvOff = recvChunk * chunkSize;
            int sendLen = Math.min(chunkSize, n - sendOff);
            int recvLen = Math.min(chunkSize, n - recvOff);
            if (sendLen <= 0 || recvLen <= 0) continue;

            sendFloats(buf, sendOff, sendLen);
            float[] recv = receiveFloats(recvLen);
            System.arraycopy(recv, 0, buf, recvOff, recvLen);
        }
        return buf;
    }

    private void sendFloats(float[] data, int offset, int len) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(len * Float.BYTES);
        for (int i = 0; i < len; i++) bb.putFloat(data[offset + i]);
        nextOut.write(bb.array());
        nextOut.flush();
    }

    private float[] receiveFloats(int len) throws IOException {
        byte[] bytes = prevIn.readNBytes(len * Float.BYTES);
        ByteBuffer bb = ByteBuffer.wrap(bytes);
        float[] result = new float[len];
        for (int i = 0; i < len; i++) result[i] = bb.getFloat();
        return result;
    }

    @Override
    public void close() throws IOException {
        if (nextSocket != null) nextSocket.close();
        serverSocket.close();
    }

    // ── Builder ───────────────────────────────────────────────────────────

    /**
     * Creates a new {@link DistributedDataParallel} builder.
     *
     * @return builder instance
     */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link DistributedDataParallel}.
     */
    public static final class Builder {
        private NNModule model;
        private int rank, worldSize;
        private String masterAddr = "localhost";
        private int masterPort = 29500;

        /** @param model the model to wrap */
        public Builder model(NNModule model)           { this.model = model; return this; }

        /** @param rank this worker's rank (0-indexed) */
        public Builder rank(int rank)                { this.rank = rank; return this; }

        /** @param worldSize total number of workers */
        public Builder worldSize(int worldSize)      { this.worldSize = worldSize; return this; }

        /** @param addr master node address (default: localhost) */
        public Builder masterAddr(String addr)       { this.masterAddr = addr; return this; }

        /** @param port base port (each rank uses masterPort + rank) */
        public Builder masterPort(int port)          { this.masterPort = port; return this; }

        /**
         * Builds and initializes the DDP wrapper, establishing ring connections.
         *
         * @return initialized {@link DistributedDataParallel}
         * @throws IOException if socket setup fails
         */
        public DistributedDataParallel build() throws IOException {
            return new DistributedDataParallel(this);
        }
    }
}
