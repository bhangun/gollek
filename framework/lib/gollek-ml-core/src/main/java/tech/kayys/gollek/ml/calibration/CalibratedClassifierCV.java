package tech.kayys.gollek.ml.calibration;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.model_selection.ModelSelection.KFold;
import tech.kayys.gollek.ml.model_selection.ModelSelection.Fold;
import java.util.*;

/**
 * Probability calibration using Platt scaling or isotonic regression.
 * Improves probability estimates for models like SVM.
 */
public class CalibratedClassifierCV extends BaseEstimator {

    private final BaseEstimator baseEstimator;
    private final String method; // "sigmoid" or "isotonic"
    private final int cv;
    private List<BaseEstimator> calibratedEstimators;
    private List<Calibrator> calibrators;

    public CalibratedClassifierCV(BaseEstimator baseEstimator, String method, int cv) {
        this.baseEstimator = baseEstimator;
        this.method = method;
        this.cv = cv;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        int nClasses = Arrays.stream(y).max().getAsInt() + 1;
        calibratedEstimators = new ArrayList<>();
        calibrators = new ArrayList<>();

        KFold kfold = new KFold(cv, true, 42);
        List<Fold> folds = kfold.split(X.length);

        for (Fold fold : folds) {
            // Split data
            float[][] XTrain = new float[fold.trainIndices.length][];
            int[] yTrain = new int[fold.trainIndices.length];
            float[][] XCal = new float[fold.valIndices.length][];
            int[] yCal = new int[fold.valIndices.length];

            for (int i = 0; i < fold.trainIndices.length; i++) {
                int idx = fold.trainIndices[i];
                XTrain[i] = X[idx];
                yTrain[i] = y[idx];
            }
            for (int i = 0; i < fold.valIndices.length; i++) {
                int idx = fold.valIndices[i];
                XCal[i] = X[idx];
                yCal[i] = y[idx];
            }

            // Train base estimator
            BaseEstimator estimator = baseEstimator.clone();
            estimator.fit(XTrain, yTrain);
            calibratedEstimators.add(estimator);

            // Get uncalibrated probabilities
            double[][] uncalibrated = estimator.predictProba(XCal);

            // Train calibrator for each class
            List<Calibrator> classCalibrators = new ArrayList<>();
            for (int c = 0; c < nClasses; c++) {
                float[] scores = new float[XCal.length];
                for (int i = 0; i < XCal.length; i++) {
                    scores[i] = (float) uncalibrated[i][c];
                }
                Calibrator calibrator = method.equals("sigmoid") ? new SigmoidCalibrator() : new IsotonicCalibrator();
                calibrator.fit(scores, yCal, c);
                classCalibrators.add(calibrator);
            }
            calibrators.addAll(classCalibrators);
        }
    }

    @Override
    public int[] predict(float[][] X) {
        double[][] probs = predictProba(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            int bestClass = 0;
            double bestProb = probs[i][0];
            for (int c = 1; c < probs[i].length; c++) {
                if (probs[i][c] > bestProb) {
                    bestProb = probs[i][c];
                    bestClass = c;
                }
            }
            predictions[i] = bestClass;
        }
        return predictions;
    }

    @Override
    public double[][] predictProba(float[][] X) {
        int nClasses = calibrators.size() / calibratedEstimators.size();
        double[][] avgProbs = new double[X.length][nClasses];

        // Average probabilities from all calibrated estimators
        for (int i = 0; i < calibratedEstimators.size(); i++) {
            BaseEstimator estimator = calibratedEstimators.get(i);
            double[][] probs = estimator.predictProba(X);

            for (int c = 0; c < nClasses; c++) {
                Calibrator calibrator = calibrators.get(i * nClasses + c);
                float[] scores = new float[X.length];
                for (int j = 0; j < X.length; j++) {
                    scores[j] = (float) probs[j][c];
                }
                float[] calibrated = calibrator.predict(scores);
                for (int j = 0; j < X.length; j++) {
                    avgProbs[j][c] += calibrated[j];
                }
            }
        }

        // Normalize
        for (int i = 0; i < X.length; i++) {
            double sum = 0;
            for (int c = 0; c < nClasses; c++) {
                avgProbs[i][c] /= calibratedEstimators.size();
                sum += avgProbs[i][c];
            }
            if (sum > 0) {
                for (int c = 0; c < nClasses; c++) {
                    avgProbs[i][c] /= sum;
                }
            }
        }

        return avgProbs;
    }

    @Override
    public boolean isFitted() {
        return calibratedEstimators != null && !calibratedEstimators.isEmpty();
    }

    // Calibrator interfaces
    interface Calibrator {
        void fit(float[] scores, int[] labels, int positiveClass);

        float[] predict(float[] scores);
    }

    /**
     * Platt scaling (sigmoid calibration).
     */
    static class SigmoidCalibrator implements Calibrator {
        private double a, b;

        @Override
        public void fit(float[] scores, int[] labels, int positiveClass) {
            // Use L-BFGS or simple gradient descent
            double[] targets = new double[labels.length];
            for (int i = 0; i < labels.length; i++) {
                targets[i] = labels[i] == positiveClass ? 1.0 : 0.0;
            }

            // Initialize parameters
            a = 0.0;
            b = 0.0;

            // Gradient descent
            double lr = 0.1;
            for (int iter = 0; iter < 100; iter++) {
                double gradA = 0, gradB = 0;
                for (int i = 0; i < scores.length; i++) {
                    double p = 1.0 / (1.0 + Math.exp(-(a * scores[i] + b)));
                    gradA += (p - targets[i]) * scores[i];
                    gradB += (p - targets[i]);
                }
                a -= lr * gradA / scores.length;
                b -= lr * gradB / scores.length;
            }
        }

        @Override
        public float[] predict(float[] scores) {
            float[] calibrated = new float[scores.length];
            for (int i = 0; i < scores.length; i++) {
                calibrated[i] = (float) (1.0 / (1.0 + Math.exp(-(a * scores[i] + b))));
            }
            return calibrated;
        }
    }

    /**
     * Isotonic regression for non-parametric calibration.
     */
    static class IsotonicCalibrator implements Calibrator {
        private List<Point> points;

        @Override
        public void fit(float[] scores, int[] labels, int positiveClass) {
            // Collect (score, target) pairs
            List<Point> data = new ArrayList<>();
            for (int i = 0; i < scores.length; i++) {
                data.add(new Point(scores[i], labels[i] == positiveClass ? 1.0 : 0.0));
            }

            // Sort by score
            data.sort((a, b) -> Float.compare(a.x, b.x));

            // PAV algorithm for isotonic regression
            List<Block> blocks = new ArrayList<>();
            for (Point p : data) {
                Block block = new Block(p.x, p.y);
                blocks.add(block);

                // Merge violating blocks
                while (blocks.size() >= 2) {
                    Block last = blocks.get(blocks.size() - 1);
                    Block prev = blocks.get(blocks.size() - 2);
                    if (prev.mean <= last.mean)
                        break;

                    prev.merge(last);
                    blocks.remove(blocks.size() - 1);
                }
            }

            // Generate points for interpolation
            points = new ArrayList<>();
            for (Block block : blocks) {
                points.add(new Point((float) block.xStart, block.mean));
                points.add(new Point((float) block.xEnd, block.mean));
            }
        }

        @Override
        public float[] predict(float[] scores) {
            float[] calibrated = new float[scores.length];
            for (int i = 0; i < scores.length; i++) {
                calibrated[i] = (float) interpolate(scores[i]);
            }
            return calibrated;
        }

        private double interpolate(float x) {
            if (x <= points.get(0).x)
                return points.get(0).y;
            if (x >= points.get(points.size() - 1).x)
                return points.get(points.size() - 1).y;

            for (int i = 0; i < points.size() - 1; i++) {
                Point p1 = points.get(i);
                Point p2 = points.get(i + 1);
                if (x >= p1.x && x <= p2.x) {
                    double t = (x - p1.x) / (p2.x - p1.x);
                    return p1.y + t * (p2.y - p1.y);
                }
            }
            return 0;
        }

        static class Point {
            float x;
            double y;

            Point(float x, double y) {
                this.x = x;
                this.y = y;
            }
        }

        static class Block {
            double xStart, xEnd;
            double sum;
            int count;
            double mean;

            Block(float x, double y) {
                this.xStart = x;
                this.xEnd = x;
                this.sum = y;
                this.count = 1;
                this.mean = y;
            }

            void merge(Block other) {
                this.xEnd = other.xEnd;
                this.sum += other.sum;
                this.count += other.count;
                this.mean = this.sum / this.count;
            }
        }
    }
}