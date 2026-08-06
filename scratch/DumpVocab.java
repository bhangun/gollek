import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;
import tech.kayys.alkhawarizm.gguf.loader.GGUFLoader;
import tech.kayys.alkhawarizm.gguf.loader.GGUFTensorInfo;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class DumpVocab {
    public static void main(String[] args) throws Exception {
        String path = "/Users/bhangun/.gollek/models/gguf/google_gemma-4-E4B-it.gguf";
        GGUFModel model = GGUFLoader.loadModel(Path.of(path));
        List<String> tokens = (List<String>) model.metadata().get("tokenizer.ggml.tokens");
        if (tokens == null)
            tokens = (List<String>) model.metadata().get("tokenizer.tokens");

        System.out.println("Vocab size: " + tokens.size());
        System.out.println("Chat Template: " + model.metadata().get("tokenizer.chat_template"));

        System.out.println("Tensors (Count: " + model.tensors().size() + "):");
        for (int i = 0; i < Math.min(20, model.tensors().size()); i++) {
            GGUFTensorInfo t = model.tensors().get(i);
            System.out.println("  " + t.name() + " " + java.util.Arrays.toString(t.shape()));
        }
    }
}

/*
 * java -cp
 * scratch:runner/native/gollek-gguf-core/target/gollek-gguf-core-0.1.0-
 * SNAPSHOT.jar:runner/native/gollek-gguf-core/target/gollek-gguf-core-0.1.0-
 * SNAPSHOT.jar DumpVocab
 */