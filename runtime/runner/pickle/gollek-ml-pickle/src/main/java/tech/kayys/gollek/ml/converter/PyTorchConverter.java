package tech.kayys.gollek.ml.converter;

import tech.kayys.gollek.ml.autograd.*;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.layer.*;
import tech.kayys.gollek.ml.pickle.*;

import java.io.IOException;
import java.util.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Complete PyTorch model converter for Gollek.
 * Supports loading .pth, .pt, and .bin files.
 */
public class PyTorchConverter {

    /**
     * Load PyTorch model from file and convert to Gollek module.
     */
    public static void loadPyTorchModel(NNModule module, String path) throws IOException {
        // Read pickle file
        byte[] data = java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(path));
        PickleParser parser = new PickleParser(data);
        PickleParser.PickleObject pickleObj = (PickleParser.PickleObject) parser.parse();

        // Extract state dict
        Map<String, Object> stateDict = extractStateDict(pickleObj);

        // Load into module
        loadStateDict(module, stateDict);
    }

    /**
     * Extract state dictionary from pickle object.
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> extractStateDict(PickleParser.PickleObject pickleObj) {
        Object state = pickleObj.getState();

        if (state instanceof Map) {
            return (Map<String, Object>) state;
        } else if (state instanceof PickleParser.PickleObject) {
            PickleParser.PickleObject stateObj = (PickleParser.PickleObject) state;
            return extractStateDict(stateObj);
        }

        return new LinkedHashMap<>();
    }

    /**
     * Load state dictionary into Gollek module.
     */
    public static void loadStateDict(NNModule module, Map<String, Object> stateDict) {
        Map<String, Parameter> params = module.namedParameters();

        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            String pytorchKey = convertKey(entry.getKey());
            Object tensor = findTensor(stateDict, pytorchKey);

            if (tensor != null) {
                float[] data = extractTensorData(tensor);
                float[] paramData = entry.getValue().data().data();

                int copyLen = Math.min(data.length, paramData.length);
                System.arraycopy(data, 0, paramData, 0, copyLen);

                System.out.println("Loaded: " + entry.getKey() + " -> " + data.length + " params");
            }
        }
    }

    /**
     * Convert Gollek parameter name to PyTorch naming convention.
     */
    private static String convertKey(String gollekKey) {
        // Gollek: "encoder.block_0.attn.weight" -> PyTorch:
        // "encoder.blocks.0.attention.weight"
        return gollekKey
                .replace("block_", "blocks.")
                .replace("attn", "attention")
                .replace("norm", "layer_norm")
                .replace("fc", "linear");
    }

    /**
     * Find tensor in state dict by key (with fuzzy matching).
     */
    private static Object findTensor(Map<String, Object> stateDict, String key) {
        // Exact match
        if (stateDict.containsKey(key)) {
            return stateDict.get(key);
        }

        // Try without module prefix
        String lastPart = key.substring(key.lastIndexOf('.') + 1);
        for (Map.Entry<String, Object> entry : stateDict.entrySet()) {
            if (entry.getKey().endsWith(lastPart)) {
                return entry.getValue();
            }
        }

        // Try partial match
        for (Map.Entry<String, Object> entry : stateDict.entrySet()) {
            if (entry.getKey().toLowerCase().contains(key.toLowerCase())) {
                return entry.getValue();
            }
        }

        return null;
    }

    /**
     * Extract float array from PyTorch tensor.
     */
    public static float[] extractTensorData(Object tensor) {
        if (tensor instanceof PickleParser.PickleObject) {
            PickleParser.PickleObject tensorObj = (PickleParser.PickleObject) tensor;
            Object data = tensorObj.getState("data");

            if (data instanceof byte[]) {
                return decodeNumpyData((byte[]) data);
            } else if (data instanceof float[]) {
                return (float[]) data;
            } else if (data instanceof double[]) {
                double[] doubles = (double[]) data;
                float[] floats = new float[doubles.length];
                for (int i = 0; i < doubles.length; i++) {
                    floats[i] = (float) doubles[i];
                }
                return floats;
            }
        } else if (tensor instanceof float[]) {
            return (float[]) tensor;
        } else if (tensor instanceof double[]) {
            double[] doubles = (double[]) tensor;
            float[] floats = new float[doubles.length];
            for (int i = 0; i < doubles.length; i++) {
                floats[i] = (float) doubles[i];
            }
            return floats;
        }

        return new float[0];
    }

    /**
     * Decode numpy array data from bytes.
     */
    private static float[] decodeNumpyData(byte[] bytes) {
        // Check if it's a numpy array (.npy format)
        if (bytes.length > 6 && bytes[0] == 0x93 && bytes[1] == 'N' &&
                bytes[2] == 'U' && bytes[3] == 'M' && bytes[4] == 'P' && bytes[5] == 'Y') {
            // Parse numpy format
            return parseNumpyArray(bytes);
        }

        // Assume raw float32 little-endian
        float[] result = new float[bytes.length / 4];
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.getFloat();
        }
        return result;
    }

    /**
     * Parse numpy .npy format array.
     */
    private static float[] parseNumpyArray(byte[] data) {
        // Find header end
        int headerEnd = findHeaderEnd(data, 10);

        // Read header
        String headerStr = new String(data, 10, headerEnd - 10);

        // Parse shape and dtype
        Map<String, Object> header = parseNumpyHeader(headerStr);
        String dtype = (String) header.get("descr");
        int[] shape = (int[]) header.get("shape");

        // Calculate total elements
        int totalElements = 1;
        for (int s : shape)
            totalElements *= s;

        // Read data
        int dataOffset = headerEnd;
        float[] result = new float[totalElements];

        if ("<f4".equals(dtype) || "<f".equals(dtype)) {
            ByteBuffer buffer = ByteBuffer.wrap(data, dataOffset, data.length - dataOffset);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < totalElements; i++) {
                result[i] = buffer.getFloat();
            }
        } else if ("<f8".equals(dtype) || "<d".equals(dtype)) {
            ByteBuffer buffer = ByteBuffer.wrap(data, dataOffset, data.length - dataOffset);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < totalElements; i++) {
                result[i] = (float) buffer.getDouble();
            }
        }

        return result;
    }

    private static int findHeaderEnd(byte[] data, int start) {
        for (int i = start; i < data.length - 1; i++) {
            if (data[i] == '\n') {
                return i + 1;
            }
        }
        return data.length;
    }

    private static Map<String, Object> parseNumpyHeader(String headerStr) {
        Map<String, Object> result = new LinkedHashMap<>();

        headerStr = headerStr.trim();
        if (headerStr.startsWith("{"))
            headerStr = headerStr.substring(1);
        if (headerStr.endsWith("}"))
            headerStr = headerStr.substring(0, headerStr.length() - 1);

        String[] parts = headerStr.split(", ");
        for (String part : parts) {
            String[] kv = part.split(": ");
            if (kv.length < 2)
                continue;

            String key = kv[0].replaceAll("'", "");
            String value = kv[1];

            if (key.equals("descr")) {
                result.put(key, value.replaceAll("'", ""));
            } else if (key.equals("fortran_order")) {
                result.put(key, Boolean.parseBoolean(value));
            } else if (key.equals("shape")) {
                value = value.replace("(", "").replace(")", "");
                String[] dims = value.split(",");
                int[] shape = new int[dims.length];
                for (int i = 0; i < dims.length; i++) {
                    String dim = dims[i].trim();
                    if (!dim.isEmpty()) {
                        shape[i] = Integer.parseInt(dim);
                    }
                }
                result.put(key, shape);
            }
        }

        return result;
    }

    /**
     * Get tensor shape from PyTorch tensor.
     */
    public static long[] getTensorShape(Object tensor) {
        if (tensor instanceof PickleParser.PickleObject) {
            PickleParser.PickleObject tensorObj = (PickleParser.PickleObject) tensor;
            Object shape = tensorObj.getState("shape");

            if (shape instanceof long[]) {
                return (long[]) shape;
            } else if (shape instanceof int[]) {
                int[] intShape = (int[]) shape;
                long[] longShape = new long[intShape.length];
                for (int i = 0; i < intShape.length; i++) {
                    longShape[i] = intShape[i];
                }
                return longShape;
            }
        }

        return new long[0];
    }
}