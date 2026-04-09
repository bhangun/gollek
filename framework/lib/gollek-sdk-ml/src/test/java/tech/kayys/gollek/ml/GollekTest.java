package tech.kayys.gollek.ml;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
// TODO: nlp package - import tech.kayys.gollek.ml.nlp.Pipeline;
import tech.kayys.gollek.runtime.tensor.Device;

import static org.junit.jupiter.api.Assertions.*;

public class GollekTest {

    @Test
    public void testTensorCreation() {
        GradTensor t = Gollek.tensor(new float[]{1, 2, 3}, 1, 3);
        assertNotNull(t);
        assertEquals(3, t.numel());
    }

    @Test
    public void testNnUsage() {
        // Use constructor directly as shown in Gollek.java examples
        Linear linear = new Linear(10, 20);
        assertNotNull(linear);
    }

    @Disabled("NLP package not yet implemented")
    @Test
    public void testPipelineFactory() {
        // Use an unknown task to trigger PipelineException
        // assertThrows(tech.kayys.gollek.ml.nlp.PipelineException.class, () -> 
        //     Gollek.pipeline("unknown-task", "any"));
    }

    @Test
    public void testDeviceHeuristics() {
        Device device = Gollek.defaultDevice();
        assertNotNull(device);
        // On most CI/Local it will be CPU
        System.out.println("Default device: " + device);
    }
}
