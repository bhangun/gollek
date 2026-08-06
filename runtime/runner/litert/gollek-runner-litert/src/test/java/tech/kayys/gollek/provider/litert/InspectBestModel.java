package tech.kayys.gollek.provider.litert;
import java.nio.file.Path;
import java.nio.file.Paths;
public class InspectBestModel {
    public static void main(String[] args) throws Exception {
        Path modelDir = Paths.get("/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm");
        java.util.Optional<Path> best = LiteRTContainerParser.findBestModelFile(modelDir);
        System.out.println("Best model: " + (best.isPresent() ? best.get() : "None"));
    }
}
