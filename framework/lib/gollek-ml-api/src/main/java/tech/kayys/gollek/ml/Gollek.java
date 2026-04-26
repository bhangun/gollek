package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.layer.ReLU;
import tech.kayys.gollek.ml.nn.layer.Dropout;
import tech.kayys.gollek.ml.nn.layer.Sequential;
import tech.kayys.gollek.ml.nn.layer.LayerNorm;
import tech.kayys.gollek.ml.nn.layer.Embedding;
import tech.kayys.gollek.ml.transformer.MultiHeadAttention;
import tech.kayys.gollek.ml.cnn.Conv2d;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.L1Loss;
import tech.kayys.gollek.ml.tensor.Tensor;
import tech.kayys.gollek.ml.tensor.Device;
import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.ml.multimodal.VisionBuilder;
import tech.kayys.gollek.ml.multimodal.MultimodalBuilder;
import tech.kayys.gollek.ml.multimodal.VideoBuilder;
import tech.kayys.gollek.ml.multimodal.AudioBuilder;
import tech.kayys.gollek.sdk.api.GollekSdk;

import tech.kayys.gollek.ml.optim.*;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.svm.*;
import tech.kayys.gollek.ml.naive_bayes.*;
import tech.kayys.gollek.ml.linear_model.*;
import tech.kayys.gollek.ml.clustering.*;
import tech.kayys.gollek.ml.pipeline.*;
import tech.kayys.gollek.ml.model_selection.*;
import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.base.BaseTransformer;
import tech.kayys.gollek.ml.util.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Main entry point for the Gollek ML framework.
 *
 * <p>
 * Provides static factory methods for creating tensors, building neural
 * networks, and querying device availability — mirroring the top-level
 * {@code torch} namespace in PyTorch, while {@code Gollek.ML} provides
 * a scikit-learn style API for traditional machine learning.
 */
public final class Gollek {

    /** Framework version. */
    public static final String VERSION = "0.1.1";

    private Gollek() {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Deep Learning (PyTorch Style)
    // ══════════════════════════════════════════════════════════════════════

    public static class DL {
        public static GradTensor tensor(float[] data, long... shape) {
            return Gollek.tensor(data, shape);
        }

        public static NNModule sequential(NNModule... layers) {
            return new Sequential(layers);
        }

        public static Optimizer adamW(List<Parameter> params, float lr) {
            return AdamW.builder(params, lr).build();
        }

        public static CrossEntropyLoss crossEntropy() {
            return new CrossEntropyLoss();
        }

        public static MSELoss mseLoss() {
            return new MSELoss();
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Traditional ML (Scikit-Learn Style)
    // ══════════════════════════════════════════════════════════════════════

    public static class ML {
        // Classification
        public static RandomForestClassifier randomForest() {
            return new RandomForestClassifier();
        }

        public static GradientBoostingClassifier gradientBoosting() {
            return new GradientBoostingClassifier();
        }

        public static SVC svm() {
            return new SVC();
        }

        public static GaussianNB naiveBayes() {
            return new GaussianNB();
        }

        public static VotingClassifier voting(List<BaseEstimator> estimators, String voting) {
            return new VotingClassifier(estimators, voting, null);
        }

        // Regression
        public static LinearModel linearRegression() {
            return new LinearModel("none", 0, 0, 1e-4, 1000, 0.01);
        }

        public static LinearModel ridge(double alpha) {
            return new LinearModel("l2", alpha, 0, 1e-4, 1000, 0.01);
        }

        public static LinearModel lasso(double alpha) {
            return new LinearModel("l1", alpha, 1, 1e-4, 1000, 0.01);
        }

        // Clustering
        public static KMeans kMeans(int nClusters) {
            return new KMeans(nClusters);
        }

        public static DBSCAN dbscan(double eps, int minSamples) {
            return new DBSCAN(eps, minSamples, "euclidean", -1);
        }

        // Preprocessing
        public static StandardScaler standardScaler() {
            return new StandardScaler();
        }

        public static PCA pca(int nComponents) {
            return new PCA(nComponents);
        }

        public static PolynomialFeatures polynomialFeatures(int degree) {
            return new PolynomialFeatures(degree);
        }

        public static Pipeline pipeline(BaseEstimator... steps) {
            if (steps.length == 0) throw new IllegalArgumentException("Pipeline must have at least one step");
            List<BaseTransformer> transformers = new ArrayList<>();
            for (int i = 0; i < steps.length - 1; i++) {
                if (!(steps[i] instanceof BaseTransformer)) {
                    throw new IllegalArgumentException("All steps except the last must be transformers");
                }
                transformers.add((BaseTransformer) steps[i]);
            }
            return new Pipeline(transformers, steps[steps.length - 1]);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Selection & Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static class Selection {
        public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y, int nFolds) {
            return CrossValidation.crossValScore(estimator, X, y, nFolds, "accuracy");
        }

        public static CrossValidation.GridSearchResult gridSearch(BaseEstimator estimator,
                Map<String, Object[]> paramGrid,
                float[][] X, int[] y) {
            return CrossValidation.gridSearch(estimator, paramGrid, X, y, 5);
        }

        public static ModelSelection.TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize, int randomState) {
            return ModelSelection.trainTestSplit(X, y, testSize, randomState);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Creation — delegates to GradTensor
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor tensor(float[] data, long... shape) {
        return GradTensor.of(data, shape);
    }

    public static GradTensor tensor(float... data) {
        return GradTensor.of(data);
    }

    public static GradTensor zeros(long... shape) {
        return GradTensor.zeros(shape);
    }

    public static GradTensor ones(long... shape) {
        return GradTensor.ones(shape);
    }

    public static GradTensor randn(long... shape) {
        return GradTensor.randn(shape);
    }

    public static GradTensor rand(long... shape) {
        return GradTensor.rand(shape);
    }

    public static GradTensor arange(float start, float end, float step) {
        return GradTensor.arange(start, end, step);
    }

    public static GradTensor arange(int end) {
        return GradTensor.arange(0, end, 1);
    }

    public static GradTensor scalar(float value) {
        return GradTensor.scalar(value);
    }

    public static GradTensor eye(int n) {
        return GradTensor.eye(n);
    }

    public static GradTensor full(float value, long... shape) {
        return GradTensor.full(value, shape);
    }

    public static GradTensor uniform(double lo, double hi, long... shape) {
        return GradTensor.uniform(lo, hi, shape);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Operations (static)
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor cat(GradTensor... tensors) {
        return GradTensor.cat(tensors);
    }

    public static GradTensor cat(int dim, GradTensor... tensors) {
        return GradTensor.cat(dim, tensors);
    }

    public static GradTensor stack(GradTensor... tensors) {
        return GradTensor.stack(tensors);
    }

    public static GradTensor stack(int dim, GradTensor... tensors) {
        return GradTensor.stack(dim, tensors);
    }

    public static GradTensor where(GradTensor condition, GradTensor x, GradTensor y) {
        return GradTensor.where(condition, x, y);
    }

    public static GradTensor einsum(String equation, GradTensor a, GradTensor b) {
        return a.einsum(equation, b);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Gradient Control
    // ══════════════════════════════════════════════════════════════════════

    public static NoGrad noGrad() {
        return NoGrad.enter();
    }

    private static GollekSdk _sdk;

    private static synchronized GollekSdk sdk() {
        if (_sdk == null) {
            _sdk = GollekSdk.builder().build();
        }
        return _sdk;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Multimodal Facade
    // ══════════════════════════════════════════════════════════════════════

    public static VisionBuilder vision(String model) {
        return new VisionBuilder(model, sdk());
    }

    public static MultimodalBuilder multimodal(String model) {
        return new MultimodalBuilder(model, sdk());
    }

    public static VideoBuilder video(String model) {
        return new VideoBuilder(model, sdk());
    }

    public static AudioBuilder audio(String model) {
        return new AudioBuilder(model, sdk());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static boolean isCudaAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        return (os.contains("linux") || os.contains("windows")) && System.getenv("CUDA_PATH") != null;
    }

    public static boolean isMetalAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        return os.contains("mac") && arch.equals("aarch64");
    }

    public static tech.kayys.gollek.runtime.tensor.Device defaultDevice() {
        if (isCudaAvailable())
            return tech.kayys.gollek.runtime.tensor.Device.CUDA;
        if (isMetalAvailable())
            return tech.kayys.gollek.runtime.tensor.Device.METAL;
        return tech.kayys.gollek.runtime.tensor.Device.CPU;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Info
    // ══════════════════════════════════════════════════════════════════════

    public static void printInfo() {
        var device = defaultDevice();
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek ML Framework                         ║");
        System.out.println("║          Version " + String.format("%-32s", VERSION) + "║");
        System.out.println("╠══════════════════════════════════════════════════════╣");
        System.out.println("║  Device:  " + String.format("%-32s", device) + "║");
        System.out.println("║  CUDA:    " + String.format("%-32s", isCudaAvailable()) + "║");
        System.out.println("║  Metal:   " + String.format("%-32s", isMetalAvailable()) + "║");
        System.out.println("║  Java:    " + String.format("%-32s", System.getProperty("java.version")) + "║");
        System.out.println("║  Vector:  "
                + String.format("%-32s", jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() + "-bit")
                + "║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
    }
}
