package tech.kayys.gollek.inference.libtorch.core;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.runtime.tensor.Tensor;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LibTorchTensorOpsTest {

    @org.junit.jupiter.api.BeforeAll
    static void init() {
        var lookup = tech.kayys.gollek.inference.libtorch.binding.NativeLibraryLoader.load(java.util.Optional.empty());
        tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.initialize(lookup);
    }

    @Test
    void testUnsqueezeAndSqueeze() {
        try (Tensor tensor = TorchTensor.zeros(new long[] { 2, 3 }, ScalarType.FLOAT, Device.CPU)) {
            assertArrayEquals(new long[] { 2, 3 }, tensor.shape());

            try (Tensor unsqueezed = tensor.unsqueeze(0)) {
                assertArrayEquals(new long[] { 1, 2, 3 }, unsqueezed.shape());

                try (tech.kayys.gollek.runtime.tensor.Tensor resqueezed = unsqueezed.squeeze()) {
                    assertArrayEquals(new long[] { 2, 3 }, resqueezed.shape());
                }
            }
        }
    }

    @Test
    void testSplit() {
        try (Tensor tensor = TorchTensor.zeros(new long[] { 10, 5 }, ScalarType.FLOAT, Device.CPU)) {
            List<tech.kayys.gollek.runtime.tensor.Tensor> chunks = tensor.split(3, 0);

            assertEquals(4, chunks.size());
            assertArrayEquals(new long[] { 3, 5 }, chunks.get(0).shape());
            assertArrayEquals(new long[] { 3, 5 }, chunks.get(1).shape());
            assertArrayEquals(new long[] { 3, 5 }, chunks.get(2).shape());
            assertArrayEquals(new long[] { 1, 5 }, chunks.get(3).shape()); // remainder

            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                chunks.forEach(c -> {
                    try {
                        c.close();
                    } catch (Exception e) {
                    }
                });
            }));
            for (tech.kayys.gollek.runtime.tensor.Tensor t : chunks) {
                t.close();
            }
        }
    }
}
