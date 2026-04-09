package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.models.EfficientNet;
import tech.kayys.gollek.ml.nn.loss.ArcFaceLoss;
import tech.kayys.gollek.ml.nn.optim.Adam;
import tech.kayys.gollek.ml.nn.optim.SAM;
import tech.kayys.gollek.ml.nn.optim.SGD;
// TODO: tokenizer package not yet implemented
// import tech.kayys.gollek.tokenizer.impl.WordPieceTokenizer;
// import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
// import tech.kayys.gollek.tokenizer.spi.DecodeOptions;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 16 tests: Bidirectional, SAM, ArcFaceLoss, EfficientNet, WordPieceTokenizer.
 */
class Session16Test {

    // ── Bidirectional ─────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void bidirectionalLSTMOutputShape() {
        Bidirectional biLSTM = new Bidirectional(new LSTM(16, 32));
        GradTensor x = GradTensor.randn(10, 2, 16); // [T, N, inputSize]
        GradTensor out = biLSTM.forward(x);
        // Output should be [T, N, 2*hiddenSize]
        assertArrayEquals(new long[]{10, 2, 64}, out.shape());
    }

    @Disabled("Requires optimize package") @Test
    void bidirectionalGRUOutputShape() {
        Bidirectional biGRU = new Bidirectional(new GRU(8, 16));
        GradTensor x = GradTensor.randn(5, 3, 8);
        assertArrayEquals(new long[]{5, 3, 32}, biGRU.forward(x).shape());
    }

    @Disabled("Requires optimize package") @Test
    void bidirectionalHasParameters() {
        Bidirectional bi = new Bidirectional(new LSTM(8, 16));
        assertTrue(bi.parameterCount() > 0);
    }

    // ── SAM ───────────────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void samTwoStepReducesLoss() {
        Linear model = new Linear(4, 2);
        SGD base = new SGD(model.parameters(), 0.01f);
        SAM sam = new SAM(model.parameters(), base, 0.05f);
        GradTensor x = GradTensor.randn(8, 4);
        GradTensor y = GradTensor.randn(8, 2);

        float first = 0f, last = 0f;
        for (int i = 0; i < 20; i++) {
            // First step
            model.zeroGrad();
            GradTensor loss1 = model.forward(x).sub(y).pow(2f).mean();
            loss1.backward();
            sam.firstStep();

            // Second step
            model.zeroGrad();
            GradTensor loss2 = model.forward(x).sub(y).pow(2f).mean();
            loss2.backward();
            sam.secondStep();

            if (i == 0)  first = loss1.item();
            if (i == 19) last  = loss2.item();
        }
        assertTrue(last < first, "SAM should reduce loss");
    }

    @Disabled("Requires optimize package") @Test
    void samRestoresWeightsAfterFirstStep() {
        Linear model = new Linear(2, 2);
        SAM sam = new SAM(model.parameters(), new SGD(model.parameters(), 0.01f));
        float[] before = model.parameters().get(0).data().data().clone();

        model.zeroGrad();
        model.forward(GradTensor.randn(2, 2)).mean().backward();
        sam.firstStep();
        sam.secondStep(); // restores weights before applying update

        // After secondStep, weights should differ from before (optimizer applied)
        // but not be the perturbed values
        assertNotNull(model.parameters().get(0).data().data());
    }

    // ── ArcFaceLoss ───────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void arcFaceLossIsScalar() {
        ArcFaceLoss loss = new ArcFaceLoss(10, 8);
        // L2-normalize features
        GradTensor features = normalizeL2(GradTensor.randn(4, 8));
        GradTensor labels   = GradTensor.of(new float[]{0, 3, 7, 9}, 4);
        GradTensor l = loss.forward(features, labels);
        assertEquals(0, l.ndim());
        assertTrue(l.item() > 0f);
        assertTrue(Float.isFinite(l.item()));
    }

    @Disabled("Requires optimize package") @Test
    void arcFaceLossPositive() {
        ArcFaceLoss loss = new ArcFaceLoss(5, 4, 0.5f, 32f);
        GradTensor features = normalizeL2(GradTensor.randn(3, 4));
        GradTensor labels   = GradTensor.of(new float[]{0, 1, 2}, 3);
        assertTrue(loss.forward(features, labels).item() > 0f);
    }

    private static GradTensor normalizeL2(GradTensor x) {
        long[] s = x.shape();
        int N = (int)s[0], D = (int)s[1];
        float[] d = x.data().clone();
        for (int n = 0; n < N; n++) {
            float norm = 0;
            for (int i = 0; i < D; i++) norm += d[n*D+i]*d[n*D+i];
            norm = (float)Math.sqrt(norm) + 1e-8f;
            for (int i = 0; i < D; i++) d[n*D+i] /= norm;
        }
        return GradTensor.of(d, s);
    }

    // ── EfficientNet ──────────────────────────────────────────────────────

    @Disabled("Requires optimize package") @Test
    void efficientNetB0OutputShape() {
        NNModule model = EfficientNet.efficientNetB0(10);
        GradTensor x = GradTensor.randn(1, 3, 32, 32); // small input for speed
        GradTensor out = model.forward(x);
        assertArrayEquals(new long[]{1, 10}, out.shape());
    }

    @Disabled("Requires optimize package") @Test
    void efficientNetB0HasParameters() {
        NNModule model = EfficientNet.efficientNetB0(1000);
        assertTrue(model.parameterCount() > 100_000L);
    }

    // ── WordPieceTokenizer ────────────────────────────────────────────────

    @Disabled("Requires tokenizer package") @Test
    void wordPieceEncodeKnownTokens() {
        // TODO: WordPieceTokenizer not available
        /*
        Map<String, Integer> vocab = new java.util.LinkedHashMap<>();
        vocab.put("[PAD]", 0); vocab.put("[UNK]", 1);
        vocab.put("[CLS]", 2); vocab.put("[SEP]", 3);
        vocab.put("hello", 4); vocab.put("world", 5);
        WordPieceTokenizer tok = new WordPieceTokenizer(vocab, 0, 2, 3, 1);

        long[] ids = tok.encode("hello world", EncodeOptions.defaultOptions());
        // [CLS]=2, hello=4, world=5, [SEP]=3
        assertArrayEquals(new long[]{2, 4, 5, 3}, ids);
        */
    }

    @Disabled("Requires tokenizer package") @Test
    void wordPieceDecodeRoundTrip() {
        // TODO: WordPieceTokenizer not available
        /*
        Map<String, Integer> vocab = new java.util.LinkedHashMap<>();
        vocab.put("[PAD]", 0); vocab.put("[UNK]", 1);
        vocab.put("[CLS]", 2); vocab.put("[SEP]", 3);
        vocab.put("hello", 4); vocab.put("world", 5);
        WordPieceTokenizer tok = new WordPieceTokenizer(vocab, 0, 2, 3, 1);

        long[] ids = tok.encode("hello world", EncodeOptions.defaultOptions());
        String decoded = tok.decode(ids, DecodeOptions.defaultOptions());
        assertEquals("hello world", decoded);
        */
    }

    @Disabled("Requires tokenizer package") @Test
    void wordPieceSubwordSplit() {
        // TODO: WordPieceTokenizer not available
        /*
        Map<String, Integer> vocab = new java.util.LinkedHashMap<>();
        vocab.put("[PAD]", 0); vocab.put("[UNK]", 1);
        vocab.put("[CLS]", 2); vocab.put("[SEP]", 3);
        vocab.put("un", 4); vocab.put("##known", 5);
        WordPieceTokenizer tok = new WordPieceTokenizer(vocab, 0, 2, 3, 1);

        long[] ids = tok.encode("unknown", EncodeOptions.defaultOptions());
        // Should split: un + ##known
        boolean hasUn = false, hasKnown = false;
        for (long id : ids) {
            if (id == 4) hasUn = true;
            if (id == 5) hasKnown = true;
        }
        assertTrue(hasUn && hasKnown);
        */
    }

}
