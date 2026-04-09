package tech.kayys.gollek.ml.distributed;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.nn.optim.Optimizer;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.List;
import java.util.function.BiFunction;

/**
 * Federated Learning trainer implementing the FedAvg algorithm.
 *
 * <p>Based on <em>"Communication-Efficient Learning of Deep Networks from
 * Decentralized Data"</em> (McMahan et al., 2017).
 *
 * <p>Each round:
 * <ol>
 *   <li>Server broadcasts global model weights to all clients</li>
 *   <li>Each client trains locally for {@code localEpochs} steps</li>
 *   <li>Clients send updated weights back to server</li>
 *   <li>Server averages all client weights (FedAvg)</li>
 * </ol>
 *
 * <p>Uses JDK 25 virtual threads for parallel client training simulation.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var federated = FederatedLearning.builder()
 *     .globalModel(serverModel)
 *     .numClients(10)
 *     .rounds(20)
 *     .localEpochs(5)
 *     .clientTrainer((model, clientId) -> trainOnClientData(model, clientId))
 *     .build();
 *
 * federated.run();
 * }</pre>
 */
public final class FederatedLearning {

    private final NNModule  globalModel;
    private final int     numClients;
    private final int     rounds;
    private final int     localEpochs;
    private final float   clientFraction; // fraction of clients per round
    private final BiFunction<NNModule, Integer, Void> clientTrainer;

    private FederatedLearning(Builder b) {
        this.globalModel     = b.globalModel;
        this.numClients      = b.numClients;
        this.rounds          = b.rounds;
        this.localEpochs     = b.localEpochs;
        this.clientFraction  = b.clientFraction;
        this.clientTrainer   = b.clientTrainer;
    }

    /**
     * Runs the full federated learning loop for {@code rounds} communication rounds.
     *
     * <p>Each round selects a random subset of clients, trains them in parallel
     * using virtual threads, then aggregates their weights via FedAvg.
     */
    public void run() {
        for (int round = 0; round < rounds; round++) {
            int selected = Math.max(1, (int) (numClients * clientFraction));
            List<Integer> clientIds = selectClients(selected);

            // Train clients in parallel using virtual threads
            List<java.util.concurrent.Future<float[][]>> futures = new java.util.ArrayList<>();
            try (var exec = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
                for (int clientId : clientIds) {
                    final int cid = clientId;
                    futures.add(exec.submit(() -> trainClient(cid)));
                }
                // Collect client weight updates
                List<float[][]> clientWeights = new java.util.ArrayList<>();
                for (var f : futures) {
                    try { clientWeights.add(f.get()); }
                    catch (Exception e) { /* skip failed client */ }
                }
                // FedAvg: average all client weights
                fedAvg(clientWeights);
            }
            System.out.printf("Round %d/%d  clients=%d%n", round + 1, rounds, selected);
        }
    }

    // ── FedAvg ────────────────────────────────────────────────────────────

    /**
     * Trains a single client: copies global weights, runs local training,
     * returns updated weight arrays.
     *
     * @param clientId client index
     * @return updated weight arrays (one per parameter)
     */
    private float[][] trainClient(int clientId) {
        // Copy global weights to a local model snapshot
        List<Parameter> params = globalModel.parameters();
        float[][] snapshot = new float[params.size()][];
        for (int i = 0; i < params.size(); i++)
            snapshot[i] = params.get(i).data().data().clone();

        // Run client-side training (user-provided)
        clientTrainer.apply(globalModel, clientId);

        // Return updated weights
        float[][] updated = new float[params.size()][];
        for (int i = 0; i < params.size(); i++)
            updated[i] = params.get(i).data().data().clone();

        // Restore global weights (so next client starts fresh)
        for (int i = 0; i < params.size(); i++)
            System.arraycopy(snapshot[i], 0, params.get(i).data().data(), 0, snapshot[i].length);

        return updated;
    }

    /**
     * FedAvg aggregation: sets global model weights to the mean of all client weights.
     * Uses {@link VectorOps} for SIMD-accelerated averaging.
     *
     * @param clientWeights list of weight arrays from each client
     */
    private void fedAvg(List<float[][]> clientWeights) {
        if (clientWeights.isEmpty()) return;
        List<Parameter> params = globalModel.parameters();
        float scale = 1.0f / clientWeights.size();

        for (int p = 0; p < params.size(); p++) {
            float[] global = params.get(p).data().data();
            java.util.Arrays.fill(global, 0f);
            for (float[][] cw : clientWeights)
                for (int i = 0; i < global.length; i++) global[i] += cw[p][i];
            VectorOps.mulScalar(global, scale, global); // SIMD scale
        }
    }

    /** Randomly selects {@code n} client IDs. */
    private List<Integer> selectClients(int n) {
        List<Integer> all = new java.util.ArrayList<>();
        for (int i = 0; i < numClients; i++) all.add(i);
        java.util.Collections.shuffle(all);
        return all.subList(0, n);
    }

    /** @return a new builder */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link FederatedLearning}.
     */
    public static final class Builder {
        private NNModule  globalModel;
        private int     numClients     = 10;
        private int     rounds         = 10;
        private int     localEpochs    = 5;
        private float   clientFraction = 0.1f;
        private BiFunction<NNModule, Integer, Void> clientTrainer;

        /** @param model the global server model */
        public Builder globalModel(NNModule m)                          { this.globalModel = m; return this; }
        /** @param n total number of clients */
        public Builder numClients(int n)                              { this.numClients = n; return this; }
        /** @param r number of communication rounds */
        public Builder rounds(int r)                                  { this.rounds = r; return this; }
        /** @param e local training epochs per round per client */
        public Builder localEpochs(int e)                             { this.localEpochs = e; return this; }
        /** @param f fraction of clients selected per round (default 0.1) */
        public Builder clientFraction(float f)                        { this.clientFraction = f; return this; }
        /** @param fn client training function: {@code (model, clientId) → null} */
        public Builder clientTrainer(BiFunction<NNModule, Integer, Void> fn) { this.clientTrainer = fn; return this; }

        /**
         * Builds the {@link FederatedLearning} trainer.
         *
         * @return configured trainer
         */
        public FederatedLearning build() { return new FederatedLearning(this); }
    }
}
