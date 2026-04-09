package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;
import tech.kayys.gollek.ml.data.DataLoader;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.optim.Optimizer;

import java.util.List;

/**
 * Knowledge Distillation trainer — trains a small student model to mimic
 * a large teacher model's soft probability distributions.
 *
 * <p>Based on <em>"Distilling the Knowledge in a Neural Network"</em> (Hinton et al., 2015).
 *
 * <p>The combined loss blends hard-label cross-entropy with soft-target KL divergence:
 * <pre>
 *   L = α · L_CE(student, labels) + (1-α) · T² · L_KL(softmax(student/T), softmax(teacher/T))
 * </pre>
 * where {@code T} is the temperature (higher = softer distributions).
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var distiller = KnowledgeDistillation.builder()
 *     .teacher(teacherModel)
 *     .student(studentModel)
 *     .optimizer(Adam.create(studentModel.parameters(), 1e-3f))
 *     .temperature(4.0f)
 *     .alpha(0.7f)
 *     .epochs(50)
 *     .build();
 *
 * distiller.fit(trainLoader);
 * }</pre>
 */
public final class KnowledgeDistillation {

    private final NNModule  teacher;
    private final NNModule  student;
    private final Optimizer optimizer;
    private final float   temperature;
    private final float   alpha;       // weight for soft loss
    private final int     epochs;

    private KnowledgeDistillation(Builder b) {
        this.teacher     = b.teacher;
        this.student     = b.student;
        this.optimizer   = b.optimizer;
        this.temperature = b.temperature;
        this.alpha       = b.alpha;
        this.epochs      = b.epochs;
    }

    /**
     * Computes the distillation loss for one batch.
     *
     * @param inputs  input batch {@code [N, ...]}
     * @param labels  hard labels {@code [N]} (class indices as float)
     * @return combined distillation loss scalar
     */
    public GradTensor distillationLoss(GradTensor inputs, GradTensor labels) {
        // Teacher forward (no grad)
        teacher.eval();
        GradTensor teacherLogits = teacher.forward(inputs).detach();

        // Student forward
        student.train();
        GradTensor studentLogits = student.forward(inputs);

        // Soft loss: KL(softmax(student/T) || softmax(teacher/T))
        GradTensor softStudent = scaledSoftmax(studentLogits, temperature);
        GradTensor softTeacher = scaledSoftmax(teacherLogits, temperature);
        GradTensor softLoss    = klDivergence(softStudent, softTeacher)
                                     .mul(temperature * temperature);

        // Hard loss: cross-entropy(student, labels)
        GradTensor hardLoss = crossEntropy(studentLogits, labels);

        // Combined: α·soft + (1-α)·hard
        return softLoss.mul(alpha).add(hardLoss.mul(1f - alpha));
    }

    /**
     * Runs the full distillation training loop.
     *
     * @param loader iterable of {@code [inputs, labels]} batches
     */
    public void fit(Iterable<DataLoader.Batch> loader) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int steps = 0;
            for (var batch : loader) {
                student.zeroGrad();
                GradTensor loss = distillationLoss(batch.inputs(), batch.labels());
                loss.backward();
                optimizer.step();
                epochLoss += loss.item();
                steps++;
            }
            System.out.printf("Epoch %d/%d  distill_loss=%.4f%n",
                epoch + 1, epochs, epochLoss / Math.max(1, steps));
        }
    }

    // ── Loss helpers ──────────────────────────────────────────────────────

    /**
     * Softmax with temperature scaling: {@code softmax(logits / T)}.
     *
     * @param logits raw logits {@code [N, C]}
     * @param T      temperature (> 0)
     * @return soft probabilities {@code [N, C]}
     */
    private static GradTensor scaledSoftmax(GradTensor logits, float T) {
        return logits.mul(1f / T).softmax();
    }

    /**
     * KL divergence: {@code Σ p * log(p/q)} averaged over batch.
     *
     * @param p student soft probabilities {@code [N, C]}
     * @param q teacher soft probabilities {@code [N, C]}
     * @return scalar KL divergence
     */
    private static GradTensor klDivergence(GradTensor p, GradTensor q) {
        // KL(p||q) = Σ p * (log p - log q)
        GradTensor logP = p.log();
        GradTensor logQ = q.log();
        return p.mul(logP.sub(logQ)).mean();
    }

    /**
     * Cross-entropy loss from logits and integer class labels.
     *
     * @param logits raw logits {@code [N, C]}
     * @param labels class indices {@code [N]}
     * @return scalar cross-entropy loss
     */
    private static GradTensor crossEntropy(GradTensor logits, GradTensor labels) {
        long[] s = logits.shape();
        int N = (int) s[0], C = (int) s[1];
        float[] lg = logits.data(), lb = labels.data();
        float[] losses = new float[N];
        for (int n = 0; n < N; n++) {
            // log-sum-exp for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++) max = Math.max(max, lg[n * C + c]);
            float sumExp = 0;
            for (int c = 0; c < C; c++) sumExp += Math.exp(lg[n * C + c] - max);
            int cls = (int) lb[n];
            losses[n] = -(lg[n * C + cls] - max - (float) Math.log(sumExp));
        }
        return GradTensor.scalar(VectorOps.sum(losses) / N);
    }

    /** @return a new builder */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link KnowledgeDistillation}.
     */
    public static final class Builder {
        private NNModule   teacher, student;
        private Optimizer optimizer;
        private float    temperature = 4.0f;
        private float    alpha       = 0.7f;
        private int      epochs      = 10;

        /** @param teacher large pre-trained teacher model */
        public Builder teacher(NNModule m)       { this.teacher = m; return this; }
        /** @param student small student model to train */
        public Builder student(NNModule m)       { this.student = m; return this; }
        /** @param opt optimizer for student parameters */
        public Builder optimizer(Optimizer o)  { this.optimizer = o; return this; }
        /** @param T temperature for softening distributions (default 4.0) */
        public Builder temperature(float T)    { this.temperature = T; return this; }
        /** @param a weight for soft loss (default 0.7); hard loss weight = 1-a */
        public Builder alpha(float a)          { this.alpha = a; return this; }
        /** @param e number of training epochs */
        public Builder epochs(int e)           { this.epochs = e; return this; }

        /**
         * Builds the {@link KnowledgeDistillation} trainer.
         *
         * @return configured trainer
         */
        public KnowledgeDistillation build()   { return new KnowledgeDistillation(this); }
    }
}
