package tech.kayys.gollek.converter.gguf;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * GGUF file validator - checks spec compliance.
 */
public final class GgufValidator {

    public static record ValidationResult(boolean valid, List<String> errors, List<String> warnings) {
    }

    public static ValidationResult validate(Path path) throws IOException {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        try (GgufReader reader = new GgufReader(path)) {
            GgufModel model = reader.read(); // Use in-memory for validation

            // Check magic
            // Already handled in reader

            // Check required metadata
            if (!model.getMeta("general.architecture").isPresent()) {
                errors.add("Missing required metadata: general.architecture");
            }

            String arch = model.architecture();

            // Check architecture-specific required keys
            if ("llama".equals(arch)) {
                checkRequiredKey(model, "llama.block_count", errors);
                checkRequiredKey(model, "llama.embedding_length", errors);
                checkRequiredKey(model, "llama.feed_forward_length", errors);
                checkRequiredKey(model, "llama.attention.head_count", errors);
                checkRequiredKey(model, "llama.context_length", errors);
            }

            // Check tensor offsets
            long dataStart = -1;
            Set<String> tensorNames = new HashSet<>();
            Map<Long, TensorInfo> offsetMap = new HashMap<>();

            for (TensorInfo ti : model.tensors()) {
                // Check duplicate names
                if (!tensorNames.add(ti.name())) {
                    errors.add("Duplicate tensor name: " + ti.name());
                }

                // Check offset alignment
                if (ti.offset() % model.alignment() != 0) {
                    errors.add("Tensor " + ti.name() + " offset " + ti.offset() +
                            " not aligned to " + model.alignment());
                }

                // Check overlapping tensors
                TensorInfo overlapping = offsetMap.get(ti.offset());
                if (overlapping != null) {
                    errors.add("Tensor " + ti.name() + " overlaps with " +
                            overlapping.name() + " at offset " + ti.offset());
                }
                offsetMap.put(ti.offset(), ti);

                // Track data start
                if (dataStart < 0 || ti.offset() < dataStart) {
                    dataStart = ti.offset();
                }
            }

            // Check tokenizer presence
            if (!model.getMeta("tokenizer.ggml.model").isPresent()) {
                warnings.add("No tokenizer metadata found");
            }

            // Check shape consistency
            model.getMeta(arch + ".embedding_length").ifPresent(embLen -> {
                long expected = embLen.asUInt32();
                model.findTensor("token_embd.weight").ifPresent(tensor -> {
                    if (tensor.ne().length > 0 && tensor.ne()[0] != expected) {
                        warnings.add("token_embd.weight dim " + tensor.ne()[0] +
                                " doesn't match embedding_length " + expected);
                    }
                });
            });
        }

        return new ValidationResult(errors.isEmpty(), errors, warnings);
    }

    private static void checkRequiredKey(GgufModel model, String key, List<String> errors) {
        if (!model.getMeta(key).isPresent()) {
            errors.add("Missing required metadata: " + key);
        }
    }
}