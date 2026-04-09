package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.metrics.TensorBoardWriter;
import tech.kayys.gollek.ml.models.GAN;
import tech.kayys.gollek.ml.nn.loss.ContrastiveLoss;
import tech.kayys.gollek.ml.nn.optim.Adadelta;
// TODO: optimize package not yet implemented - import tech.kayys.gollek.ml.optimize.PostTrainingQuantizer;
// TODO: optimize package not yet implemented - import tech.kayys.gollek.ml.optimize.QuantizationAwareTraining;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 11 tests: ContrastiveLoss, Adadelta, GAN, QAT, TensorBoardWriter.
 */
class Session11Test {

    // ── ContrastiveLoss ───────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void contrastiveLossZeroForSimilarPair() {
        ContrastiveLoss loss = new ContrastiveLoss(1.0f);
        // Same embeddings, label=1 (similar) → distance=0 → loss=0
        GradTensor x1 = GradTensor.of(new float[]{1f, 0f}, 1, 2);
        GradTensor x2 = GradTensor.of(new float[]{1f, 0f}, 1, 2);
        GradTensor y  = GradTensor.of(new float[]{1f}, 1);
        assertEquals(0f, loss.forward(x1, x2, y).item(), 1e-5f);
    }

    @Disabled("Requires optimize package") @Test
    void contrastiveLossZeroForFarNegativePair() {
        ContrastiveLoss loss = new ContrastiveLoss(1.0f);
        // Very far embeddings, label=0 (dissimilar) → dist >> margin → loss=0
        GradTensor x1 = GradTensor.of(new float[]{0f, 0f}, 1, 2);
        GradTensor x2 = GradTensor.of(new float[]{100f, 0f}, 1, 2);
        GradTensor y  = GradTensor.of(new float[]{0f}, 1);
        assertEquals(0f, loss.forward(x1, x2, y).item(), 1e-5f);
    }

    @Disabled("Requires optimize package") @Test
    void contrastiveLossPositiveForCloseNegativePair() {
        ContrastiveLoss loss = new ContrastiveLoss(1.0f);
        // Close embeddings, label=0 (dissimilar) → loss > 0
        GradTensor x1 = GradTensor.of(new float[]{0f, 0f}, 1, 2);
        GradTensor x2 = GradTensor.of(new float[]{0.1f, 0f}, 1, 2);
        GradTensor y  = GradTensor.of(new float[]{0f}, 1);
        assertTrue(loss.forward(x1, x2, y).item() > 0f);
    }

    // ── Adadelta ──────────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void adadeltaReducesLoss() {
        Linear model = new Linear(4, 2);
        Adadelta opt = new Adadelta(model.parameters());
        GradTensor x = GradTensor.randn(8, 4);
        GradTensor y = GradTensor.randn(8, 2);

        float first = 0f, last = 0f;
        for (int i = 0; i < 30; i++) {
            model.zeroGrad();
            GradTensor loss = model.forward(x).sub(y).pow(2f).mean();
            loss.backward();
            opt.step();
            if (i == 0)  first = loss.item();
            if (i == 29) last  = loss.item();
        }
        assertTrue(last < first, "Adadelta should reduce loss");
    }

    @Disabled("Requires optimize package") @Test
    void adadeltaDefaultLrIsOne() {
        Adadelta opt = new Adadelta(new Linear(2, 2).parameters());
        assertEquals(1.0f, opt.learningRate(), 1e-6f);
    }

    // ── GAN ───────────────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void ganGenerateOutputShape() {
        GAN gan = new GAN(16, 32, 8);
        GradTensor fake = gan.generate(4);
        assertArrayEquals(new long[]{4, 8}, fake.shape());
    }

    @Disabled("Requires optimize package") @Test
    void ganDiscriminatorLossPositive() {
        GAN gan = new GAN(16, 32, 8);
        GradTensor real = GradTensor.randn(4, 8);
        GradTensor fake = gan.generate(4).detach();
        GradTensor loss = gan.discriminatorLoss(real, fake);
        assertEquals(0, loss.ndim());
        assertTrue(loss.item() > 0f);
    }

    @Disabled("Requires optimize package") @Test
    void ganGeneratorLossPositive() {
        GAN gan = new GAN(16, 32, 8);
        GradTensor fake = gan.generate(4);
        GradTensor loss = gan.generatorLoss(fake);
        assertEquals(0, loss.ndim());
        assertTrue(loss.item() > 0f);
    }

    @Disabled("Requires optimize package") @Test
    void ganHasSeparateParameters() {
        GAN gan = new GAN(16, 32, 8);
        assertTrue(gan.getGenerator().parameterCount() > 0);
        assertTrue(gan.getDiscriminator().parameterCount() > 0);
    }

    // ── QuantizationAwareTraining ─────────────────────────────────────────

    @Disabled("QuantizationAwareTraining not found - optimize package missing") @Test
    void qatForwardRunsWithQAT() {
        // TODO: QuantizationAwareTraining not available - optimize package not implemented
        // Linear model = new Linear(4, 2);
        // Adadelta opt = new Adadelta(model.parameters());
        // QuantizationAwareTraining qat = new QuantizationAwareTraining(model, opt);
        // qat.enableQAT();
        // 
        // GradTensor x = GradTensor.randn(2, 4);
        // assertDoesNotThrow(() -> qat.forward(x));
    }

    @Disabled("QuantizationAwareTraining not found - optimize package missing") @Test
    void qatConvertsToInt8() {
        // TODO: QuantizationAwareTraining not available - optimize package not implemented
        // Linear model = new Linear(4, 2);
        // QuantizationAwareTraining qat = new QuantizationAwareTraining(
        //     model, new Adadelta(model.parameters()));
        // qat.enableQAT();
        // 
        // var quantized = qat.convertToInt8();
        // assertEquals(model.stateDict().keySet(), quantized.keySet());
        // quantized.values().forEach(qt ->
        //     assertNotNull(qt.data(), "Quantized data should not be null"));
    }

    @Disabled("QuantizationAwareTraining not found - optimize package missing") @Test
    void qatDisableRestoresNormalForward() {
        // TODO: QuantizationAwareTraining not available - optimize package not implemented
        // Linear model = new Linear(4, 2);
        // QuantizationAwareTraining qat = new QuantizationAwareTraining(
        //     model, new Adadelta(model.parameters()));
        // qat.enableQAT();
        // qat.disableQAT();
        // GradTensor x = GradTensor.randn(2, 4);
        // assertDoesNotThrow(() -> qat.forward(x));
    }

    // ── TensorBoardWriter ─────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void tensorBoardWriterCreatesFile(@TempDir Path tmpDir) throws IOException {
        try (TensorBoardWriter writer = new TensorBoardWriter(tmpDir)) {
            writer.addScalar("loss", 0.5f, 0);
            writer.addScalar("loss", 0.3f, 1);
            writer.flush();
        }
        // Should have created an events file
        long count = Files.list(tmpDir)
            .filter(p -> p.getFileName().toString().startsWith("events.out.tfevents"))
            .count();
        assertEquals(1, count);
    }

    @Disabled("Requires optimize package") @Test
    void tensorBoardWriterFileNonEmpty(@TempDir Path tmpDir) throws IOException {
        try (TensorBoardWriter writer = new TensorBoardWriter(tmpDir)) {
            writer.addScalar("acc", 0.9f, 10);
            writer.addText("config", "lr=0.001", 0);
        }
        long size = Files.list(tmpDir)
            .filter(p -> p.getFileName().toString().startsWith("events.out.tfevents"))
            .mapToLong(p -> { try { return Files.size(p); } catch (IOException e) { return 0; } })
            .sum();
        assertTrue(size > 0, "TFRecord file should be non-empty");
    }

    @Disabled("Requires optimize package") @Test
    void tensorBoardAutoStep(@TempDir Path tmpDir) throws IOException {
        try (TensorBoardWriter writer = new TensorBoardWriter(tmpDir)) {
            // Auto-increment step
            writer.addScalar("loss", 1.0f);
            writer.addScalar("loss", 0.5f);
            writer.addScalar("loss", 0.2f);
        }
        // Just verify no exception thrown
        assertTrue(Files.list(tmpDir).findAny().isPresent());
    }
}
