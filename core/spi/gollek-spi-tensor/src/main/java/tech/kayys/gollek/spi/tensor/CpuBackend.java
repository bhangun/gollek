package tech.kayys.gollek.spi.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Default pure-Java CPU implementation of {@link ComputeBackend}.
 *
 * <p>All operations run on the JVM heap using standard Java arrays.
 * This backend is always available and serves as the fallback when
 * no hardware-accelerated backend is on the classpath.
 */
public final class CpuBackend implements ComputeBackend {

    @Override
    public float[] matmul(float[] a, long[] shapeA, float[] b, long[] shapeB) {
        int m = (int) shapeA[shapeA.length - 2];
        int k = (int) shapeA[shapeA.length - 1];
        int n = (int) shapeB[shapeB.length - 1];

        // Compute batch size statically avoiding arrays out-of-bounds on pure 2D tensors
        int batchA = 1, batchB = 1;
        for (int i = 0; i < shapeA.length - 2; i++) batchA *= (int) shapeA[i];
        for (int i = 0; i < shapeB.length - 2; i++) batchB *= (int) shapeB[i];
        int batch = Math.max(batchA, batchB);

        float[] result = new float[batch * m * n];
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        int loopBound = SPECIES.loopBound(n);

        // Vectorized SIMD batched multiplication via parallel row processing
        for (int bIdx = 0; bIdx < batch; bIdx++) {
            final int aOff = (bIdx % batchA) * m * k;
            final int bOff = (bIdx % batchB) * k * n;
            final int rOff = bIdx * m * n;
            
            java.util.stream.IntStream.range(0, m).parallel().forEach(i -> {
                for (int p = 0; p < k; p++) {
                    float a_ip = a[aOff + i * k + p];
                    int j = 0;
                    for (; j < loopBound; j += SPECIES.length()) {
                        FloatVector rVec = FloatVector.fromArray(SPECIES, result, rOff + i * n + j);
                        FloatVector bVec = FloatVector.fromArray(SPECIES, b, bOff + p * n + j);
                        rVec.add(bVec.mul(a_ip)).intoArray(result, rOff + i * n + j);
                    }
                    for (; j < n; j++) {
                        result[rOff + i * n + j] += a_ip * b[bOff + p * n + j];
                    }
                }
            });
        }
        return result;
    }

    @Override
    public float[] add(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int bound = species.loopBound(a.length);
        int i = 0;
        for (; i < bound; i += species.length()) {
            FloatVector.fromArray(species, a, i).add(FloatVector.fromArray(species, b, i)).intoArray(c, i);
        }
        for (; i < a.length; i++) c[i] = a[i] + b[i];
        return c;
    }

    @Override
    public float[] sub(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int bound = species.loopBound(a.length);
        int i = 0;
        for (; i < bound; i += species.length()) {
            FloatVector.fromArray(species, a, i).sub(FloatVector.fromArray(species, b, i)).intoArray(c, i);
        }
        for (; i < a.length; i++) c[i] = a[i] - b[i];
        return c;
    }

    @Override
    public float[] mul(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int bound = species.loopBound(a.length);
        int i = 0;
        for (; i < bound; i += species.length()) {
            FloatVector.fromArray(species, a, i).mul(FloatVector.fromArray(species, b, i)).intoArray(c, i);
        }
        for (; i < a.length; i++) c[i] = a[i] * b[i];
        return c;
    }

    @Override
    public float[] div(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int bound = species.loopBound(a.length);
        int i = 0;
        for (; i < bound; i += species.length()) {
            FloatVector.fromArray(species, a, i).div(FloatVector.fromArray(species, b, i)).intoArray(c, i);
        }
        for (; i < a.length; i++) c[i] = a[i] / b[i];
        return c;
    }

    @Override
    public float[] relu(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.max(0, data[i]);
        }
        return result;
    }

    @Override
    public float[] sigmoid(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) (1.0 / (1.0 + Math.exp(-data[i])));
        }
        return result;
    }

    @Override
    public float[] tanh(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.tanh(data[i]);
        }
        return result;
    }

    @Override
    public float[] exp(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.exp(data[i]);
        }
        return result;
    }

    @Override
    public float[] log(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.log(data[i]);
        }
        return result;
    }

    @Override
    public float[] sum(float[] data, long[] shape) {
        float s = 0;
        for (float v : data) s += v;
        return new float[]{s};
    }

    @Override
    public float[] mean(float[] data, long[] shape) {
        float s = 0;
        for (float v : data) s += v;
        return new float[]{s / data.length};
    }

    @Override
    public float[] transpose2d(float[] data, long rows, long cols) {
        int r = (int) rows, c = (int) cols;
        float[] result = new float[data.length];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                result[j * r + i] = data[i * c + j];
            }
        }
        return result;
    }

    @Override
    public float[] pow(float[] data, long[] shape, float p) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.pow(data[i], p);
        }
        return result;
    }

    @Override
    public String deviceName() {
        return "CPU";
    }

    @Override
    public int priority() {
        return 0; // lowest — any GPU backend takes precedence
    }
}
