package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.util.SafetensorReader;
import tech.kayys.gollek.ml.nn.util.SafetensorWriter;
import tech.kayys.gollek.ml.nn.util.gguf.GgufReader;
import tech.kayys.gollek.ml.nn.util.gguf.GgufWriter;

import java.io.*;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

/**
 * Base class for all neural network modules in Gollek ML.
 * <p>
 * Every layer, model, or network component extends {@code Module}.
 * Modules form a tree: a module can contain sub-modules, creating
 * a hierarchy that mirrors the neural network architecture.
 *
 * <h3>Defining a custom module</h3>
 * <pre>{@code
 * public class MyModel extends NNModule {
 *     private final Linear fc1 = register(new Linear(784, 128));
 *     private final Linear fc2 = register(new Linear(128, 10));
 *
 *     @Override
 *     public GradTensor forward(GradTensor input) {
 *         var x = fc1.forward(input).relu();
 *         return fc2.forward(x);
 *     }
 * }
 * }</pre>
 *
 * <h3>PyTorch equivalence</h3>
 * This class is the Gollek equivalent of {@code torch.nn.Module}.
 */
public abstract class NNModule {

    private boolean training = true;
    private final Map<String, Parameter> parameters = new LinkedHashMap<>();
    private final Map<String, NNModule> children = new LinkedHashMap<>();

    // ── Forward ──────────────────────────────────────────────────────────

    /**
     * Defines the computation performed at every call.
     *
     * @param input the input tensor
     * @return the output tensor
     */
    public abstract GradTensor forward(GradTensor input);

    // ── Parameter management ─────────────────────────────────────────────

    /**
     * Register a parameter with a name.
     * Parameters are trainable tensors owned by this module.
     */
    protected Parameter registerParameter(String name, GradTensor data) {
        Parameter param = new Parameter(data.requiresGrad(true));
        parameters.put(name, param);
        return param;
    }

    /**
     * Register a child module. Returns the child for assignment.
     */
    protected <T extends NNModule> T register(String name, T child) {
        children.put(name, child);
        return child;
    }

    /**
     * Register a child module with auto-generated name.
     */
    protected <T extends NNModule> T register(T child) {
        String name = child.getClass().getSimpleName().toLowerCase(Locale.ROOT) + "_" + children.size();
        return register(name, child);
    }

    /**
     * Get all parameters of this module (including sub-modules).
     */
    public List<Parameter> parameters() {
        List<Parameter> all = new ArrayList<>(parameters.values());
        for (NNModule child : children.values()) {
            all.addAll(child.parameters());
        }
        return Collections.unmodifiableList(all);
    }

    /**
     * Get named parameters as a map (dot-separated paths for nested modules).
     */
    public Map<String, Parameter> namedParameters() {
        return namedParameters("");
    }

    private Map<String, Parameter> namedParameters(String prefix) {
        Map<String, Parameter> result = new LinkedHashMap<>();
        for (var entry : parameters.entrySet()) {
            String key = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            result.put(key, entry.getValue());
        }
        for (var entry : children.entrySet()) {
            String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            result.putAll(entry.getValue().namedParameters(childPrefix));
        }
        return result;
    }

    /**
     * Get all child modules.
     */
    public Map<String, NNModule> modules() {
        Map<String, NNModule> all = new LinkedHashMap<>();
        all.put("", this);
        for (var entry : children.entrySet()) {
            all.put(entry.getKey(), entry.getValue());
            for (var sub : entry.getValue().modules().entrySet()) {
                String key = sub.getKey().isEmpty() ? entry.getKey() : entry.getKey() + "." + sub.getKey();
                all.put(key, sub.getValue());
            }
        }
        return all;
    }

    // ── Training/Eval mode ───────────────────────────────────────────────

    /** Set the module (and all sub-modules) to training mode. */
    public NNModule train() {
        this.training = true;
        children.values().forEach(NNModule::train);
        return this;
    }

    /** Set the module (and all sub-modules) to evaluation mode. */
    public NNModule eval() {
        this.training = false;
        children.values().forEach(NNModule::eval);
        return this;
    }

    /** Check if in training mode. */
    public boolean isTraining() {
        return training;
    }

    // ── Zero grad ────────────────────────────────────────────────────────

    /** Zero all gradients. Call before each training step. */
    public void zeroGrad() {
        for (Parameter p : parameters()) {
            p.data().zeroGrad();
        }
    }

    // ── Parameter count ──────────────────────────────────────────────────

    /** Total number of trainable parameters. */
    public long parameterCount() {
        return parameters().stream().mapToLong(p -> p.data().numel()).sum();
    }

    /** Human-readable parameter count. */
    public String parameterCountFormatted() {
        long count = parameterCount();
        if (count >= 1_000_000_000) return String.format("%.1fB", count / 1e9);
        if (count >= 1_000_000) return String.format("%.1fM", count / 1e6);
        if (count >= 1_000) return String.format("%.1fK", count / 1e3);
        return String.valueOf(count);
    }

    // ── Serialization ────────────────────────────────────────────────────

    /** Save model weights to a file. */
    public void save(Path path) throws IOException {
        Map<String, float[]> stateDict = new LinkedHashMap<>();
        for (var entry : namedParameters().entrySet()) {
            stateDict.put(entry.getKey(), entry.getValue().data().data().clone());
        }
        try (var oos = new ObjectOutputStream(new FileOutputStream(path.toFile()))) {
            oos.writeObject(stateDict);
        }
    }

    /** Save model weights in the universal Safetensor format. */
    public void saveSafetensors(Path path) throws IOException {
        Map<String, GradTensor> stateDict = new LinkedHashMap<>();
        for (var entry : namedParameters().entrySet()) {
            stateDict.put(entry.getKey(), entry.getValue().data());
        }
        SafetensorWriter.save(path, stateDict);
    }

    /** Save model weights in the universal Safetensor format. */
    public void saveSafetensors(String path) throws IOException {
        saveSafetensors(Path.of(path));
    }

    /** Load model weights from the universal Safetensor format. */
    public void loadSafetensors(Path path) throws IOException {
        Map<String, float[]> stateDict = SafetensorReader.read(path);
        Map<String, Parameter> params = namedParameters();
        
        for (var entry : stateDict.entrySet()) {
            Parameter param = params.get(entry.getKey());
            if (param != null) {
                float[] data = entry.getValue();
                if (data.length == param.data().numel()) {
                    System.arraycopy(data, 0, param.data().data(), 0, data.length);
                } else {
                    System.err.println("Warning: Shape mismatch for " + entry.getKey() + 
                                       ". Expected " + param.data().numel() + " elements, got " + data.length);
                }
            } else {
                System.err.println("Warning: Parameter " + entry.getKey() + " not found in model.");
            }
        }
    }

    /** Load model weights from the universal Safetensor format. */
    public void loadSafetensors(String path) throws IOException {
        loadSafetensors(Path.of(path));
    }

    /** Save model weights in the GGUF format. */
    public void saveGguf(Path path) throws IOException {
        Map<String, GradTensor> stateDict = new LinkedHashMap<>();
        for (var entry : namedParameters().entrySet()) {
            stateDict.put(entry.getKey(), entry.getValue().data());
        }
        GgufWriter.save(path, stateDict, Map.of("general.architecture", tech.kayys.gollek.ml.nn.util.gguf.GgufMetaValue.ofString("gollek")));
    }

    /** Save model weights in the GGUF format. */
    public void saveGguf(String path) throws IOException {
        saveGguf(Path.of(path));
    }

    /** Load model weights from the GGUF format. */
    public void loadGguf(Path path) throws IOException {
        try (GgufReader reader = new GgufReader(path)) {
            Map<String, float[]> stateDict = reader.loadTensors();
            Map<String, Parameter> params = namedParameters();

            for (var entry : stateDict.entrySet()) {
                Parameter param = params.get(entry.getKey());
                if (param != null) {
                    float[] data = entry.getValue();
                    if (data.length == param.data().numel()) {
                        System.arraycopy(data, 0, param.data().data(), 0, data.length);
                    } else {
                        System.err.println("Warning: Shape mismatch for " + entry.getKey() +
                                           ". Expected " + param.data().numel() + " elements, got " + data.length);
                    }
                } else {
                    System.err.println("Warning: Parameter " + entry.getKey() + " not found in model.");
                }
            }
        }
    }

    /** Load model weights from the GGUF format. */
    public void loadGguf(String path) throws IOException {
        loadGguf(Path.of(path));
    }

    /** Load model weights from a file. */
    @SuppressWarnings("unchecked")
    public void load(Path path) throws IOException, ClassNotFoundException {
        Map<String, float[]> stateDict;
        try (var ois = new ObjectInputStream(new FileInputStream(path.toFile()))) {
            stateDict = (Map<String, float[]>) ois.readObject();
        }
        Map<String, Parameter> params = namedParameters();
        for (var entry : stateDict.entrySet()) {
            Parameter param = params.get(entry.getKey());
            if (param != null) {
                System.arraycopy(entry.getValue(), 0, param.data().data(), 0, entry.getValue().length);
            }
        }
    }

    // ── StateDict (PyTorch-compatible API) ───────────────────────────────

    /**
     * Returns a map of parameter name → tensor (shallow copy of data references).
     * Compatible with {@link StateDict#save} / {@link StateDict#load}.
     */
    public Map<String, GradTensor> stateDict() {
        Map<String, GradTensor> sd = new LinkedHashMap<>();
        namedParameters().forEach((name, param) -> sd.put(name, param.data()));
        return sd;
    }

    /**
     * Load parameters from a state dict. Strict mode (default): all keys must match.
     */
    public void loadStateDict(Map<String, GradTensor> state) {
        loadStateDict(state, true);
    }

    /**
     * Load parameters from a state dict.
     *
     * @param strict if true, throws on missing/unexpected keys; if false, loads what matches
     */
    public void loadStateDict(Map<String, GradTensor> state, boolean strict) {
        Map<String, Parameter> params = namedParameters();
        for (var entry : state.entrySet()) {
            Parameter p = params.get(entry.getKey());
            if (p == null) {
                if (strict) throw new IllegalArgumentException(
                    "loadStateDict: unexpected key '" + entry.getKey() + "'");
                continue;
            }
            float[] src = entry.getValue().data();
            float[] dst = p.data().data();
            if (src.length != dst.length) {
                if (strict) throw new IllegalArgumentException(
                    "loadStateDict: size mismatch for '" + entry.getKey() +
                    "': expected " + dst.length + ", got " + src.length);
                continue;
            }
            System.arraycopy(src, 0, dst, 0, src.length);
        }
    }

    // ── String representation ────────────────────────────────────────────

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName()).append("(\n");
        for (var entry : children.entrySet()) {
            sb.append("  (").append(entry.getKey()).append("): ");
            String childStr = entry.getValue().toString().replace("\n", "\n  ");
            sb.append(childStr).append("\n");
        }
        sb.append(")");
        return sb.toString();
    }
}
