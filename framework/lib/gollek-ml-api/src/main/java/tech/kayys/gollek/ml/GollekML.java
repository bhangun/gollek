package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.svm.*;
import tech.kayys.gollek.ml.naive_bayes.*;
import tech.kayys.gollek.ml.linear_model.*;
import tech.kayys.gollek.ml.clustering.*;
import tech.kayys.gollek.ml.pipeline.*;
import tech.kayys.gollek.ml.model_selection.*;
import tech.kayys.gollek.ml.util.*;

import java.util.List;
import java.util.Map;

/**
 * @deprecated Use {@link Gollek} as the unified ML entry point.
 *             {@code Gollek.ML} mirrors the scikit-learn factory API and
 *             {@code Gollek.DL} mirrors the PyTorch/autograd API.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class GollekML {

    // ==================== Deep Learning ====================
    public static class DL {
        public static GradTensor tensor(float[] data, long... shape) {
            return Gollek.tensor(data, shape);
        }

        public static NNModule sequential(NNModule... layers) {
            return Gollek.DL.sequential(layers);
        }

        public static Optimizer adamW(List<Parameter> params, float lr) {
            return Gollek.DL.adamW(params, lr);
        }

        public static CrossEntropyLoss crossEntropy() {
            return Gollek.DL.crossEntropy();
        }
    }

    // ==================== Traditional ML ====================
    public static class ML {
        // Classification
        public static RandomForestClassifier randomForest() {
            return Gollek.ML.randomForest();
        }

        public static GradientBoostingClassifier gradientBoosting() {
            return Gollek.ML.gradientBoosting();
        }

        public static SVC svm() {
            return Gollek.ML.svm();
        }

        public static GaussianNB naiveBayes() {
            return Gollek.ML.naiveBayes();
        }

        // Regression
        public static LinearModel linearRegression() {
            return Gollek.ML.linearRegression();
        }

        public static LinearModel ridge(double alpha) {
            return Gollek.ML.ridge(alpha);
        }

        public static LinearModel lasso(double alpha) {
            return Gollek.ML.lasso(alpha);
        }

        // Clustering
        public static KMeans kMeans(int nClusters) {
            return Gollek.ML.kMeans(nClusters);
        }

        public static DBSCAN dbscan(double eps, int minSamples) {
            return Gollek.ML.dbscan(eps, minSamples);
        }

        // Preprocessing
        public static StandardScaler standardScaler() {
            return Gollek.ML.standardScaler();
        }

        public static PCA pca(int nComponents) {
            return Gollek.ML.pca(nComponents);
        }

        public static PolynomialFeatures polynomialFeatures(int degree) {
            return Gollek.ML.polynomialFeatures(degree);
        }
    }

    // ==================== Model Selection ====================
    public static class Selection {
        public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y, int nFolds) {
            return Gollek.Selection.crossValScore(estimator, X, y, nFolds);
        }

        public static CrossValidation.GridSearchResult gridSearch(BaseEstimator estimator,
                Map<String, Object[]> paramGrid,
                float[][] X, int[] y) {
            return Gollek.Selection.gridSearch(estimator, paramGrid, X, y);
        }

        public static ModelSelection.TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize, int randomState) {
            return Gollek.Selection.trainTestSplit(X, y, testSize, randomState);
        }
    }
}