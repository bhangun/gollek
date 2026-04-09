package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Graph Convolutional Network (GCN) layer — equivalent to {@code torch_geometric.nn.GCNConv}.
 *
 * <p>Based on <em>"Semi-Supervised Classification with Graph Convolutional Networks"</em>
 * (Kipf & Welling, 2017).
 *
 * <p>Propagation rule:
 * <pre>
 *   H' = σ(D̃⁻¹/² Ã D̃⁻¹/² H W)
 *   where Ã = A + I  (adjacency with self-loops)
 *         D̃ = degree matrix of Ã
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var gcn = new GCNConv(inFeatures=16, outFeatures=32);
 * // x: [N, inFeatures], adj: [N, N] adjacency matrix
 * GradTensor out = gcn.forward(x, adj); // [N, outFeatures]
 * }</pre>
 */
public final class GCNConv extends NNModule {

    private final Linear linear;

    /**
     * Creates a GCN convolution layer.
     *
     * @param inFeatures  input node feature dimension
     * @param outFeatures output node feature dimension
     */
    public GCNConv(int inFeatures, int outFeatures) {
        this.linear = register("linear", new Linear(inFeatures, outFeatures, false));
    }

    /**
     * Forward pass: normalized graph convolution.
     *
     * @param x   node features {@code [N, inFeatures]}
     * @param adj adjacency matrix {@code [N, N]} (can be weighted, no self-loops needed)
     * @return updated node features {@code [N, outFeatures]}
     */
    public GradTensor forward(GradTensor x, GradTensor adj) {
        int N = (int) x.shape()[0];
        float[] a = adj.data();

        // Add self-loops: Ã = A + I
        float[] aHat = a.clone();
        for (int i = 0; i < N; i++) aHat[i * N + i] += 1f;

        // Compute degree: D̃[i] = sum of row i
        float[] deg = new float[N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) deg[i] += aHat[i * N + j];

        // Symmetric normalization: D̃⁻¹/² Ã D̃⁻¹/²
        float[] norm = new float[N * N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float di = deg[i] > 0 ? (float)(1.0 / Math.sqrt(deg[i])) : 0f;
                float dj = deg[j] > 0 ? (float)(1.0 / Math.sqrt(deg[j])) : 0f;
                norm[i * N + j] = di * aHat[i * N + j] * dj;
            }

        // Aggregate: normAdj @ x  [N, inFeatures]
        float[] xd = x.data();
        int F = (int) x.shape()[1];
        float[] aggr = VectorOps.matmul(norm, xd, N, N, F);

        // Linear transform + ReLU
        return linear.forward(GradTensor.of(aggr, N, F)).relu();
    }

    @Override
    public GradTensor forward(GradTensor x) {
        throw new UnsupportedOperationException("GCNConv requires adjacency matrix. Use forward(x, adj).");
    }

    @Override public String toString() { return "GCNConv(" + linear + ")"; }
}
