package tech.kayys.gollek.ml.ensemble;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.tree.DecisionTreeRegressor;
import java.util.*;

/**
 * Gradient Boosting Machine - sequential ensemble of weak learners.
 */
public class GradientBoostingClassifier extends BaseEstimator {

    private final List<DecisionTreeRegressor> trees = new ArrayList<>();
    private final List<Double> treeWeights = new ArrayList<>();
    private final int nEstimators;
    private final double learningRate;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final String loss; // deviance, exponential
    private final double subsample;
    private final int earlyStoppingRounds;

    private int nClasses;
    private double initPredictionValue;
    private double[] initPredictions;
    private boolean isFitted = false;

    public GradientBoostingClassifier() {
        this(100, 0.1, 3, 2, "deviance", 1.0, 10);
    }

    public GradientBoostingClassifier(int nEstimators, double learningRate,
            int maxDepth, int minSamplesSplit,
            String loss, double subsample,
            int earlyStoppingRounds) {
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.loss = loss;
        this.subsample = subsample;
        this.earlyStoppingRounds = earlyStoppingRounds;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        int nSamples = X.length;
        this.nClasses = Arrays.stream(y).max().getAsInt() + 1;

        // Initialize predictions with log-odds
        double mean = Arrays.stream(y).average().orElse(0.5);
        this.initPredictionValue = Math.log(mean / (1.0 - mean + 1e-11) + 1e-11);
        
        double[] currentPredictions = new double[nSamples];
        Arrays.fill(currentPredictions, initPredictionValue);

        double lastLoss = Double.MAX_VALUE;
        int roundsNoImprove = 0;

        Random rng = new Random(42);

        for (int round = 0; round < nEstimators; round++) {
            // Negative gradient of log loss
            double[] residuals = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                double p = 1.0 / (1.0 + Math.exp(-currentPredictions[i]));
                residuals[i] = y[i] - p;
            }

            // Subsampling
            int[] indices = getSubsampleIndices(nSamples, subsample, rng);
            float[][] Xsub = new float[indices.length][];
            double[] rsub = new double[indices.length];
            for (int i = 0; i < indices.length; i++) {
                Xsub[i] = X[indices[i]];
                rsub[i] = residuals[indices[i]];
            }

            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, minSamplesSplit);
            tree.fit(Xsub, rsub);

            // Update predictions
            for (int i = 0; i < nSamples; i++) {
                currentPredictions[i] += learningRate * tree.predictSingleValue(X[i]);
            }

            trees.add(tree);
            treeWeights.add(learningRate);

            // Early stopping check
            double currentLoss = computeLoss(y, currentPredictions);
            if (currentLoss < lastLoss - 1e-6) {
                lastLoss = currentLoss;
                roundsNoImprove = 0;
            } else {
                roundsNoImprove++;
            }

            if (earlyStoppingRounds > 0 && roundsNoImprove >= earlyStoppingRounds) break;
        }

        this.isFitted = true;
    }

    private int[] getSubsampleIndices(int n, double fraction, Random rng) {
        if (fraction >= 1.0) {
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            return idx;
        }
        int k = (int) (n * fraction);
        int[] idx = new int[k];
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) list.add(i);
        Collections.shuffle(list, rng);
        for (int i = 0; i < k; i++) idx[i] = list.get(i);
        return idx;
    }

    private double computeLoss(int[] y, double[] predictions) {
        double loss = 0;
        for (int i = 0; i < y.length; i++) {
            double p = 1.0 / (1.0 + Math.exp(-predictions[i]));
            loss -= y[i] * Math.log(p + 1e-11) + (1 - y[i]) * Math.log(1.0 - p + 1e-11);
        }
        return loss / y.length;
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double pred = initPredictionValue;
            for (int t = 0; t < trees.size(); t++) {
                pred += treeWeights.get(t) * trees.get(t).predictSingleValue(X[i]);
            }
            predictions[i] = pred > 0 ? 1 : 0;
        }
        return predictions;
    }

    @Override
    public double[][] predictProba(float[][] X) {
        validateInput(X);
        double[][] probs = new double[X.length][2];
        for (int i = 0; i < X.length; i++) {
            double pred = initPredictionValue;
            for (int t = 0; t < trees.size(); t++) {
                pred += treeWeights.get(t) * trees.get(t).predictSingleValue(X[i]);
            }
            double p = 1.0 / (1.0 + Math.exp(-pred));
            probs[i][1] = p;
            probs[i][0] = 1.0 - p;
        }
        return probs;
    }

    public void addTree(DecisionTreeRegressor tree, double weight) {
        this.trees.add(tree);
        this.treeWeights.add(weight);
        this.isFitted = true;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }
}