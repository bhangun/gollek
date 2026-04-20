package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.data.DataLoader;
import tech.kayys.gollek.ml.data.Dataset;
import tech.kayys.gollek.ml.metrics.MetricsTracker;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.WarmupCosineScheduler;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

import java.util.List;

/**
 * Text classification fine-tuning trainer — wraps a pre-trained encoder
 * with a classification head and trains it on labeled text data.
 *
 * <p>Implements the standard BERT fine-tuning recipe:
 * AdamW + warmup cosine schedule + cross-entropy loss.
 */
public final class TextClassificationTrainer {

    private final NNModule  model;
    private final Tokenizer tokenizer;
    private final Optimizer optimizer;
    private final int       epochs;
    private final int       warmupSteps;
    private final int       maxLength;
    private final int       batchSize;
    private final MetricsTracker metrics = new MetricsTracker();

    private TextClassificationTrainer(Builder b) {
        this.model       = b.model;
        this.tokenizer   = b.tokenizer;
        this.optimizer   = b.optimizer;
        this.epochs      = b.epochs;
        this.warmupSteps = b.warmupSteps;
        this.maxLength   = b.maxLength;
        this.batchSize   = b.batchSize;
    }

    /**
     * Fine-tunes the model on text classification data.
     */
    public void fit(List<String> trainTexts, List<Integer> trainLabels,
                    List<String> valTexts,   List<Integer> valLabels) {

        int totalBatches = (int) Math.ceil((double) trainTexts.size() / batchSize);
        int totalSteps = totalBatches * epochs;
        WarmupCosineScheduler scheduler = new WarmupCosineScheduler(
            optimizer, warmupSteps, totalSteps, optimizer.learningRate(), 0f);

        CrossEntropyLoss lossFn = new CrossEntropyLoss();

        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float trainLoss = 0f; int trainSteps = 0;

            for (int i = 0; i < trainTexts.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainTexts.size());
                List<String> batchTexts  = trainTexts.subList(i, end);
                List<Integer> batchLabels = trainLabels.subList(i, end);

                GradTensor x = encodeTexts(batchTexts);
                GradTensor y = encodeLabels(batchLabels);

                model.zeroGrad();
                GradTensor logits = model.forward(x);
                GradTensor loss   = lossFn.compute(logits, y);
                loss.backward();
                optimizer.step();
                scheduler.step();

                trainLoss += loss.item();
                trainSteps++;
            }

            float avgTrain = trainSteps > 0 ? trainLoss / trainSteps : 0f;
            metrics.log("train/loss", avgTrain, epoch);

            if (valTexts != null && !valTexts.isEmpty()) {
                float valAcc = evaluate(valTexts, valLabels);
                metrics.log("val/accuracy", valAcc, epoch);
                System.out.printf("Epoch %d/%d  train_loss=%.4f  val_acc=%.4f%n",
                    epoch+1, epochs, avgTrain, valAcc);
            } else {
                System.out.printf("Epoch %d/%d  train_loss=%.4f%n", epoch+1, epochs, avgTrain);
            }
        }
    }

    public float evaluate(List<String> texts, List<Integer> labels) {
        model.eval();
        int correct = 0;
        for (int i = 0; i < texts.size(); i += batchSize) {
            int end = Math.min(i + batchSize, texts.size());
            GradTensor x = encodeTexts(texts.subList(i, end));
            GradTensor logits = model.forward(x);
            float[] d = logits.data();
            int N = end - i, C = (int)(logits.numel() / N);
            for (int n = 0; n < N; n++) {
                int pred = 0;
                for (int c = 1; c < C; c++) if (d[n*C+c] > d[n*C+pred]) pred = c;
                if (pred == labels.get(i + n)) correct++;
            }
        }
        return (float) correct / texts.size();
    }

    public MetricsTracker metrics() { return metrics; }

    // ── Encoding helpers ──────────────────────────────────────────────────

    private GradTensor encodeTexts(List<String> texts) {
        int N = texts.size();
        float[] ids = new float[N * maxLength];
        int padId = tokenizer.padTokenId();
        
        EncodeOptions options = EncodeOptions.defaultOptions();
        
        for (int n = 0; n < N; n++) {
            long[] tokens = tokenizer.encode(texts.get(n), options);
            for (int t = 0; t < maxLength; t++) {
                ids[n * maxLength + t] = t < tokens.length ? (float) tokens[t] : (float) padId;
            }
        }
        return GradTensor.of(ids, N, maxLength);
    }

    private static GradTensor encodeLabels(List<Integer> labels) {
        float[] d = new float[labels.size()];
        for (int i = 0; i < labels.size(); i++) d[i] = labels.get(i);
        return GradTensor.of(d, labels.size());
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private NNModule  model;
        private Tokenizer tokenizer;
        private Optimizer optimizer;
        private int       epochs      = 3;
        private int       warmupSteps = 100;
        private int       maxLength   = 128;
        private int       batchSize   = 32;

        public Builder model(NNModule m)       { this.model = m; return this; }
        public Builder tokenizer(Tokenizer t){ this.tokenizer = t; return this; }
        public Builder optimizer(Optimizer o){ this.optimizer = o; return this; }
        public Builder epochs(int e)         { this.epochs = e; return this; }
        public Builder warmupSteps(int w)    { this.warmupSteps = w; return this; }
        public Builder maxLength(int m)      { this.maxLength = m; return this; }
        public Builder batchSize(int b)      { this.batchSize = b; return this; }
        public TextClassificationTrainer build() { return new TextClassificationTrainer(this); }
    }
}
