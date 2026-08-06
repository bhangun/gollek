import tech.kayys.alkhawarizm.gguf.loader.GGUFLoader;
import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public final class GgufInspect {
    private static final List<String> KEYS = List.of(
            "general.name",
            "general.architecture",
            "general.basename",
            "general.finetune",
            "general.type",
            "general.quantization_version",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token",
            "gemma2.context_length",
            "gemma2.embedding_length",
            "gemma2.block_count",
            "gemma2.attention.head_count",
            "gemma2.attention.head_count_kv",
            "gemma2.attention.key_length",
            "gemma2.attention.layer_norm_rms_epsilon",
            "gemma2.rope.freq_base",
            "gemma2.attn_logit_softcapping",
            "gemma2.final_logit_softcapping",
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.attention.key_length",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.rope.freq_base"
    );

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: GgufInspect <path-to-model.gguf>");
            System.exit(1);
        }

        Path modelPath = Path.of(args[0]);
        try (GGUFModel model = GGUFLoader.loadModel(modelPath)) {
            Map<String, Object> meta = model.metadata();
            System.out.println("model=" + modelPath);
            System.out.println("metadata.count=" + meta.size());
            System.out.println("tensor.count=" + model.tensors().size());

            for (String key : KEYS) {
                Object value = meta.get(key);
                if (value != null) {
                    System.out.println(key + "=" + format(value));
                }
            }

            dumpTokens(meta, "tokenizer.ggml.tokens");
            dumpTokens(meta, "tokenizer.ggml.token_type");
        }
    }

    @SuppressWarnings("unchecked")
    private static void dumpTokens(Map<String, Object> meta, String key) {
        Object value = meta.get(key);
        if (!(value instanceof List<?> list)) {
            return;
        }
        System.out.println(key + ".size=" + list.size());
        int limit = Math.min(24, list.size());
        for (int i = 0; i < limit; i++) {
            System.out.println(key + "[" + i + "]=" + format(list.get(i)));
        }
    }

    private static String format(Object value) {
        if (value instanceof byte[] bytes) {
            return Arrays.toString(bytes);
        }
        if (value instanceof List<?> list) {
            return "List(size=" + list.size() + ")";
        }
        return String.valueOf(value);
    }
}
