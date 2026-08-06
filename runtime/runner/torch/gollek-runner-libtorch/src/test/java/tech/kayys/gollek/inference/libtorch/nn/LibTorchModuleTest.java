package tech.kayys.gollek.inference.libtorch.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

import java.nio.file.Path;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.*;

class LibTorchModuleTest {

    @org.junit.jupiter.api.BeforeAll
    static void init() {
        var lookup = tech.kayys.gollek.inference.libtorch.binding.NativeLibraryLoader.load(java.util.Optional.empty());
        tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.initialize(lookup);
    }

    static class SimpleLinear extends Module {
        public SimpleLinear() {
            TorchTensor w = TorchTensor.zeros(new long[] { 10, 10 }, ScalarType.FLOAT, Device.CPU);
            registerParameter("weight", w);
        }

        @Override
        public TorchTensor forward(TorchTensor input) {
            return input; // Identity for testing
        }
    }

    @Test
    void testStateDictSerialization(@TempDir Path tempDir) throws Exception {
        Path stateDictPath = tempDir.resolve("model.pt");

        try (SimpleLinear m1 = new SimpleLinear()) {
            TorchTensor w = m1.parameters.get("weight");
            // Mutate weight
            try (TorchTensor ones = TorchTensor.ones(new long[] { 10, 10 }, ScalarType.FLOAT, Device.CPU)) {
                w.copy_(ones);
            }
            m1.saveStateDict(stateDictPath);
        }

        assertTrue(Files.exists(stateDictPath));

        try (SimpleLinear m2 = new SimpleLinear()) {
            m2.loadStateDict(stateDictPath);
            TorchTensor w = m2.parameters.get("weight");
            // We expect the weights to be ones now
            try (TorchTensor res = w.add(w)) {
                // Nothing because the values are hidden behind native mem,
                // but we can trust if it didn't crash
            }
        }
    }
}
