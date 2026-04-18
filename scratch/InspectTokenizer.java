import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

public class InspectTokenizer {
    public static void main(String[] args) throws Exception {
        String path = "/Users/bhangun/.gollek/models/gguf/Qwen_Qwen2.5-0.5B-Instruct.gguf";
        GGUFModel model = GGUFLoader.loadModel(Paths.get(path));
        Map<String, Object> meta = model.metadata();
        List<String> tokens = (List<String>) meta.get("tokenizer.ggml.tokens");
        if (tokens == null) tokens = (List<String>) meta.get("tokenizer.tokens");
        
        System.out.println("Vocab size: " + tokens.size());
        for (int i = 0; i < 500; i++) {
            String t = tokens.get(i);
            if (t.contains(" ") || t.contains("\u2581") || t.contains("\u0120") || t.length() > 1) {
                System.out.println("Token [" + i + "]: '" + t + "' (hex: " + toHex(t) + ")");
            }
        }
    }
    
    private static String toHex(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            sb.append(String.format("\\u%04x", (int) c));
        }
        return sb.toString();
    }
}
