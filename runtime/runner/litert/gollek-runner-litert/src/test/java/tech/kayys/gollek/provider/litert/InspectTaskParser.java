package tech.kayys.gollek.provider.litert;

import java.nio.file.Path;
import java.nio.file.Paths;

public class InspectTaskParser {
    public static void main(String[] args) throws Exception {
        String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.task";
        Path p = Paths.get(pathStr);
        LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(p);
        System.out.println("Format: " + info.format());
        System.out.println("isLlm: " + info.isLlmModel());
        System.out.println("Offset: " + info.tfliteOffset());
        System.out.println("Size: " + info.tfliteSize());
        System.out.println("SubModels: " + info.subModels().size());
        for (var sub : info.subModels()) {
            System.out.println(" - " + sub.modelType() + " at " + sub.offset() + " size " + sub.size());
        }
        System.out.println("Metadata buffers:");
        for (var entry : info.metadataBuffers().entrySet()) {
            System.out.println("  " + entry.getKey() + " -> offset: " + entry.getValue().dataOffset() + ", size: "
                    + entry.getValue().dataSize());
        }
    }
}
