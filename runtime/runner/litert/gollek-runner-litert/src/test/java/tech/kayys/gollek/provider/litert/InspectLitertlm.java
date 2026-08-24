package tech.kayys.gollek.provider.litert;
import java.nio.file.Path;
import java.nio.file.Paths;
public class InspectLitertlm {
    public static void main(String[] args) throws Exception {
        Path p = Paths.get("/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm");
        LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(p);
        System.out.println("isLlm: " + info.isLlmModel());
        System.out.println("SubModels: " + info.subModels().size());
        for (var sub : info.subModels()) {
            System.out.println(" - " + sub.modelType() + " at " + sub.offset() + " size " + sub.size());
        }
    }
}
