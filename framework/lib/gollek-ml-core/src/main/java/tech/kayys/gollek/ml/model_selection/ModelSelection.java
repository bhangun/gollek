package tech.kayys.gollek.ml.model_selection;

import tech.kayys.gollek.ml.base.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

/**
 * Cross-validation and model selection utilities.
 */
public class ModelSelection {

    /**
     * K-Fold cross-validation splitter.
     */
    public static class KFold {
        private final int nSplits;
        private final boolean shuffle;
        private final int randomState;

        public KFold(int nSplits) {
            this(nSplits, false, 42);
        }

        public KFold(int nSplits, boolean shuffle, int randomState) {
            this.nSplits = nSplits;
            this.shuffle = shuffle;
            this.randomState = randomState;
        }

        public List<Fold> split(int nSamples) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < nSamples; i++)
                indices.add(i);

            if (shuffle) {
                Collections.shuffle(indices, new Random(randomState));
            }

            List<Fold> folds = new ArrayList<>();
            int foldSize = nSamples / nSplits;

            for (int fold = 0; fold < nSplits; fold++) {
                int start = fold * foldSize;
                int end = (fold == nSplits - 1) ? nSamples : start + foldSize;

                List<Integer> trainIdx = new ArrayList<>();
                List<Integer> valIdx = new ArrayList<>();

                for (int i = 0; i < nSamples; i++) {
                    if (i >= start && i < end) {
                        valIdx.add(indices.get(i));
                    } else {
                        trainIdx.add(indices.get(i));
                    }
                }

                folds.add(new Fold(
                        trainIdx.stream().mapToInt(i -> i).toArray(),
                        valIdx.stream().mapToInt(i -> i).toArray()));
            }

            return folds;
        }
    }

    /**
     * Stratified K-Fold - preserves class distribution.
     */
    public static class StratifiedKFold {
        private final int nSplits;
        private final int randomState;

        public StratifiedKFold(int nSplits, int randomState) {
            this.nSplits = nSplits;
            this.randomState = randomState;
        }

        public List<Fold> split(float[][] X, int[] y) {
            // Count samples per class
            int nClasses = Arrays.stream(y).max().getAsInt() + 1;
            List<List<Integer>> classIndices = new ArrayList<>();
            for (int c = 0; c < nClasses; c++) {
                classIndices.add(new ArrayList<>());
            }

            for (int i = 0; i < y.length; i++) {
                classIndices.get(y[i]).add(i);
            }

            // Shuffle each class
            Random rng = new Random(randomState);
            for (List<Integer> indices : classIndices) {
                Collections.shuffle(indices, rng);
            }

            // Split each class proportionally
            List<List<Integer>> trainFolds = new ArrayList<>();
            List<List<Integer>> valFolds = new ArrayList<>();
            for (int i = 0; i < nSplits; i++) {
                trainFolds.add(new ArrayList<>());
                valFolds.add(new ArrayList<>());
            }

            for (List<Integer> classIdx : classIndices) {
                int foldSize = classIdx.size() / nSplits;
                for (int fold = 0; fold < nSplits; fold++) {
                    int start = fold * foldSize;
                    int end = (fold == nSplits - 1) ? classIdx.size() : start + foldSize;

                    for (int j = 0; j < classIdx.size(); j++) {
                        if (j >= start && j < end) {
                            valFolds.get(fold).add(classIdx.get(j));
                        } else {
                            trainFolds.get(fold).add(classIdx.get(j));
                        }
                    }
                }
            }

            List<Fold> result = new ArrayList<>();
            for (int i = 0; i < nSplits; i++) {
                result.add(new Fold(
                    trainFolds.get(i).stream().mapToInt(idx -> idx).toArray(),
                    valFolds.get(i).stream().mapToInt(idx -> idx).toArray()
                ));
            }
            return result;
        }
    }

    /**
     * Train-test split utility.
     */
    public static class TrainTestSplit {
        public final float[][] XTrain;
        public final float[][] XTest;
        public final int[] yTrain;
        public final int[] yTest;

        public TrainTestSplit(float[][] XTrain, float[][] XTest, int[] yTrain, int[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }

    public static TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize, int randomState) {
        int nSamples = X.length;
        int nTest = (int) (nSamples * testSize);

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++)
            indices.add(i);
        Collections.shuffle(indices, new Random(randomState));

        float[][] XTest = new float[nTest][];
        int[] yTest = new int[nTest];
        float[][] XTrain = new float[nSamples - nTest][];
        int[] yTrain = new int[nSamples - nTest];

        for (int i = 0; i < nSamples; i++) {
            int idx = indices.get(i);
            if (i < nTest) {
                XTest[i] = X[idx];
                yTest[i] = y[idx];
            } else {
                XTrain[i - nTest] = X[idx];
                yTrain[i - nTest] = y[idx];
            }
        }

        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }

    public static class Fold {
        public final int[] trainIndices;
        public final int[] valIndices;

        public Fold(int[] trainIndices, int[] valIndices) {
            this.trainIndices = trainIndices;
            this.valIndices = valIndices;
        }
    }
}