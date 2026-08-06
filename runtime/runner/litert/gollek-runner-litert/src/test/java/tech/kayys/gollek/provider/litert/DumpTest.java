package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.Test;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

public class DumpTest {

    @Test
    public void dumpArchDimensions() throws Exception {
        Path taskPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "litert", "litert-community", "gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it-web.task");
        
        Map<String, LiteRTContainerParser.WeightEntry> weights = LiteRTContainerParser.extractWeightMap(taskPath);
        
        System.out.println("=== LAYERS WITH K PROJECTION ===");
        for (int i = 0; i < 35; i++) {
            boolean hasK = weights.containsKey("transformer.layer_" + i + ".attn.k.w");
            boolean hasV = weights.containsKey("transformer.layer_" + i + ".attn.v.w");
            System.out.println("Layer " + i + ": hasK=" + hasK + ", hasV=" + hasV);
        }
    }
}
