package tech.kayys.gollek.ml.serialize;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/**
 * Bridge for NumPy array serialization.
 * Allows direct conversion between Gollek tensors and NumPy arrays.
 */
public class NumPyBridge {

    /**
     * Save float array as NumPy .npy file.
     */
    public static void saveAsNumpy(float[] array, String path) throws IOException {
        saveAsNumpy(new float[][] { array }, path);
    }

    /**
     * Save 2D float array as NumPy .npy file.
     */
    public static void saveAsNumpy(float[][] matrix, String path) throws IOException {
        int rows = matrix.length;
        int cols = matrix[0].length;
        
        // Build numpy header
        ByteArrayOutputStream header = new ByteArrayOutputStream();
        DataOutputStream headerOut = new DataOutputStream(header);
        
        // Magic string
        headerOut.writeByte(0x93);
        headerOut.writeBytes("NUMPY");
        headerOut.writeByte(1); // Major version
        headerOut.writeByte(0); // Minor version
        
        // Header dict
        Map<String, Object> headerDict = new LinkedHashMap<>();
        headerDict.put("descr", "<f4"); // Little-endian float32
        headerDict.put("fortran_order", false);
        headerDict.put("shape", new int[]{rows, cols});
        
        String headerStr = headerDictToString(headerDict);
        int headerLen = headerStr.length();
        
        // Pad to multiple of 64 bytes
        int paddedLen = ((headerLen + 1 + 64 - 1) / 64) * 64;
        headerOut.writeBytes(headerStr);
        for (int i = headerLen; i < paddedLen - 1; i++) {
            headerOut.writeByte(' ');
        }
        headerOut.writeByte('\n');
        
        // Write header
        try (FileOutputStream fos = new FileOutputStream(path)) {
            fos.write(header.toByteArray());
            
            // Write data in row-major order, little-endian
            ByteBuffer buffer = ByteBuffer.allocate(rows * cols * 4);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            for (float[] row : matrix) {
                for (float v : row) {
                    buffer.putFloat(v);
                }
            }
            fos.write(buffer.array());
        }
    }

    /**
     * Load NumPy .npy file.
     */
    public static float[][] loadFromNumpy(String path) throws IOException {
        byte[] data = Files.readAllBytes(Paths.get(path));

        // Check magic
        if (data[0] != (byte) 0x93 || data[1] != 'N' || data[2] != 'U' ||
                data[3] != 'M' || data[4] != 'P' || data[5] != 'Y') {
            throw new IOException("Invalid .npy file");
        }

        int version = data[6] & 0xFF;
        if (version != 1) {
            throw new IOException("Unsupported .npy version: " + version);
        }

        // Find header end
        int headerEnd = findHeaderEnd(data, 10);
        String headerStr = new String(data, 10, headerEnd - 10);

        // Parse header
        Map<String, Object> header = parseHeader(headerStr);
        String dtype = (String) header.get("descr");
        int[] shape = (int[]) header.get("shape");

        if (!"<f4".equals(dtype)) {
            throw new IOException("Unsupported dtype: " + dtype + " (only float32 supported)");
        }

        int rows = shape[0];
        int cols = shape.length > 1 ? shape[1] : 1;
        float[][] matrix = new float[rows][cols];

        // Read data
        ByteBuffer buffer = ByteBuffer.wrap(data, headerEnd, data.length - headerEnd);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = buffer.getFloat();
            }
        }

        return matrix;
    }

    /**
     * Convert float array to NumPy format bytes.
     */
    public static byte[] toNumpyBytes(float[] array) {
        int rows = 1;
        int cols = array.length;
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        
        try {
            // Magic and version
            out.writeByte(0x93);
            out.writeBytes("NUMPY");
            out.writeByte(1);
            out.writeByte(0);
            
            // Header
            Map<String, Object> header = new LinkedHashMap<>();
            header.put("descr", "<f4");
            header.put("fortran_order", false);
            header.put("shape", new int[]{rows, cols});
            
            String headerStr = headerDictToString(header);
            int headerLen = headerStr.length();
            int paddedLen = ((headerLen + 1 + 64 - 1) / 64) * 64;
            
            out.writeBytes(headerStr);
            for (int i = headerLen; i < paddedLen - 1; i++) {
                out.writeByte(' ');
            }
            out.writeByte('\n');
            
            // Data
            ByteBuffer buffer = ByteBuffer.allocate(array.length * 4);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            for (float v : array) {
                buffer.putFloat(v);
            }
            out.write(buffer.array());
            
            return baos.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int findHeaderEnd(byte[] data, int start) {
        for (int i = start; i < data.length - 1; i++) {
            if (data[i] == '\n') {
                return i + 1;
            }
        }
        throw new RuntimeException("Invalid .npy file: no header end found");
    }

    private static String headerDictToString(Map<String, Object> header) {
        StringBuilder sb = new StringBuilder("{");
        int i = 0;
        for (Map.Entry<String, Object> entry : header.entrySet()) {
            if (i++ > 0)
                sb.append(", ");
            sb.append("'").append(entry.getKey()).append("': ");

            if (entry.getValue() instanceof String) {
                sb.append("'").append(entry.getValue()).append("'");
            } else if (entry.getValue() instanceof int[]) {
                sb.append(arrayToString((int[]) entry.getValue()));
            } else if (entry.getValue() instanceof boolean) {
                sb.append(entry.getValue());
            } else {
                sb.append(entry.getValue());
            }
        }
        sb.append("}");
        return sb.toString();
    }

    private static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0)
                sb.append(", ");
            sb.append(arr[i]);
        }
        if (arr.length == 1)
            sb.append(",");
        sb.append(")");
        return sb.toString();
    }

    private static Map<String, Object> parseHeader(String headerStr) {
        Map<String, Object> result = new LinkedHashMap<>();

        // Remove braces and split
        headerStr = headerStr.trim();
        if (headerStr.startsWith("{"))
            headerStr = headerStr.substring(1);
        if (headerStr.endsWith("}"))
            headerStr = headerStr.substring(0, headerStr.length() - 1);

        String[] parts = headerStr.split(", ");
        for (String part : parts) {
            String[] kv = part.split(": ");
            String key = kv[0].replaceAll("'", "");
            String value = kv[1];

            if (key.equals("descr")) {
                result.put(key, value.replaceAll("'", ""));
            } else if (key.equals("fortran_order")) {
                result.put(key, Boolean.parseBoolean(value));
            } else if (key.equals("shape")) {
                // Parse tuple like "(2, 3)" or "(2,)"
                value = value.replace("(", "").replace(")", "");
                String[] dims = value.split(",");
                int[] shape = new int[dims.length];
                for (int i = 0; i < dims.length; i++) {
                    String dim = dims[i].trim();
                    if (dim.isEmpty())
                        continue;
                    shape[i] = Integer.parseInt(dim);
                }
                if (shape.length == 0)
                    shape = new int[] { 1 };
                result.put(key, shape);
            }
        }

        return result;
    }
}