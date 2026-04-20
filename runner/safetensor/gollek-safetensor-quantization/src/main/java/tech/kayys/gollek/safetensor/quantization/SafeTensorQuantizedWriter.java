/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafeTensorQuantizedWriter.java
 * ───────────────────────
 * SafeTensor writer for quantized models.
 */
package tech.kayys.gollek.safetensor.quantization;

import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Writer for quantized SafeTensor models.
 * <p>
 * SafeTensors is a binary format for storing tensors that is:
 * <ul>
 * <li>Safe: No arbitrary code execution during loading</li>
 * <li>Fast: Memory-mappable for zero-copy loading</li>
 * <li>Simple: Straightforward binary structure</li>
 * </ul>
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class SafeTensorQuantizedWriter {

    private static final Logger log = Logger.getLogger(SafeTensorQuantizedWriter.class);

    /**
     * SafeTensors file header magic number.
     */
    private static final long HEADER_SIZE_OFFSET = 0;

    /**
     * Write quantized model to SafeTensors format.
     *
     * @param outputPath    output file path
     * @param quantizedData map of tensor name to quantized data
     * @param tensorInfos   tensor metadata
     * @param config        quantization configuration
     * @throws IOException if write fails
     */
    public void write(Path outputPath, Map<String, byte[]> quantizedData,
            Map<String, QuantizedTensorInfo> tensorInfos, QuantConfig config) throws IOException {
        log.infof("Writing quantized SafeTensors model to: %s", outputPath);

        try (FileChannel channel = FileChannel.open(outputPath, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            // Build header
            Map<String, Object> header = buildHeader(quantizedData, tensorInfos, config);
            byte[] headerBytes = serializeHeader(header);

            // Write header size (8 bytes, little-endian)
            ByteBuffer sizeBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            sizeBuffer.putLong(headerBytes.length);
            sizeBuffer.flip();
            channel.write(sizeBuffer);

            // Write header
            channel.write(ByteBuffer.wrap(headerBytes));

            // Write tensor data
            long offset = 8 + headerBytes.length;
            for (Map.Entry<String, byte[]> entry : quantizedData.entrySet()) {
                String name = entry.getKey();
                byte[] data = entry.getValue();

                // Write tensor data
                ByteBuffer dataBuffer = ByteBuffer.wrap(data);
                channel.write(dataBuffer);

                log.debugf("Wrote tensor %s (%d bytes) at offset %d", name, data.length, offset);
                offset += data.length;
            }
        }

        log.infof("Successfully wrote quantized model with %d tensors", quantizedData.size());
    }

    /**
     * Write single quantized tensor.
     *
     * @param outputPath output file path
     * @param name       tensor name
     * @param data       quantized data
     * @param info       tensor metadata
     * @throws IOException if write fails
     */
    public void writeTensor(Path outputPath, String name, byte[] data, QuantizedTensorInfo info) throws IOException {
        Map<String, byte[]> quantizedData = new HashMap<>();
        quantizedData.put(name, data);

        Map<String, QuantizedTensorInfo> tensorInfos = new HashMap<>();
        tensorInfos.put(name, info);

        write(outputPath, quantizedData, tensorInfos, QuantConfig.builder().build());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Build SafeTensors header.
     *
     * @param quantizedData quantized tensor data
     * @param tensorInfos   tensor metadata
     * @param config        quantization configuration
     * @return header map
     */
    private Map<String, Object> buildHeader(Map<String, byte[]> quantizedData,
            Map<String, QuantizedTensorInfo> tensorInfos, QuantConfig config) {
        Map<String, Object> header = new LinkedHashMap<>();

        long offset = 0;
        for (Map.Entry<String, byte[]> entry : quantizedData.entrySet()) {
            String name = entry.getKey();
            byte[] data = entry.getValue();
            QuantizedTensorInfo info = tensorInfos.get(name);

            Map<String, Object> tensorSpec = new LinkedHashMap<>();

            if (info != null) {
                // Shape
                tensorSpec.put("shape", info.getShape());

                // Data type
                tensorSpec.put("dtype", info.getQuantizedDtype());

                // Quantization metadata
                Map<String, Object> quantSpec = new LinkedHashMap<>();
                quantSpec.put("strategy", info.getStrategy().name());
                quantSpec.put("group_size", info.getGroupSize());
                quantSpec.put("channel_axis", info.getChannelAxis());

                if (info.getScales() != null && info.getScales().length > 0) {
                    quantSpec.put("scales", info.getScales());
                }
                if (info.getZeros() != null && info.getZeros().length > 0) {
                    quantSpec.put("zeros", info.getZeros());
                }

                tensorSpec.put("quantization", quantSpec);
            } else {
                // Default spec
                tensorSpec.put("shape", new int[] { data.length });
                tensorSpec.put("dtype", "U8");
            }

            // Data offsets
            tensorSpec.put("data_offsets", new long[] { offset, offset + data.length });

            header.put(name, tensorSpec);
            offset += data.length;
        }

        // Add global metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("format", "safetensors");
        metadata.put("version", "1.0");
        metadata.put("quantization_config", buildQuantizationConfig(config));
        header.put("__metadata__", metadata);

        return header;
    }

    /**
     * Build quantization config metadata.
     *
     * @param config quantization configuration
     * @return config map
     */
    private Map<String, Object> buildQuantizationConfig(QuantConfig config) {
        Map<String, Object> quantConfig = new LinkedHashMap<>();
        quantConfig.put("strategy", config.getStrategy().name());
        quantConfig.put("bits", config.getBits());
        quantConfig.put("group_size", config.getGroupSize());
        quantConfig.put("symmetric", config.isSymmetric());
        quantConfig.put("per_channel", config.isPerChannel());
        quantConfig.put("act_order", config.isActOrder());
        quantConfig.put("damp_percent", config.getDampPercent());
        return quantConfig;
    }

    /**
     * Serialize header to JSON bytes.
     *
     * @param header header map
     * @return JSON bytes
     * @throws IOException if serialization fails
     */
    private byte[] serializeHeader(Map<String, Object> header) throws IOException {
        // Simple JSON serialization (in production, use a proper JSON library)
        StringBuilder json = new StringBuilder();
        json.append("{");

        boolean first = true;
        for (Map.Entry<String, Object> entry : header.entrySet()) {
            if (!first) {
                json.append(",");
            }
            first = false;

            json.append("\"").append(escapeJson(entry.getKey())).append("\":");
            serializeJsonValue(json, entry.getValue());
        }

        json.append("}");
        return json.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * Serialize JSON value.
     *
     * @param sb    string builder
     * @param value value to serialize
     * @throws IOException if serialization fails
     */
    @SuppressWarnings("unchecked")
    private void serializeJsonValue(StringBuilder sb, Object value) throws IOException {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String) {
            sb.append("\"").append(escapeJson((String) value)).append("\"");
        } else if (value instanceof Number) {
            sb.append(value.toString());
        } else if (value instanceof Boolean) {
            sb.append(value.toString());
        } else if (value instanceof int[]) {
            int[] arr = (int[]) value;
            sb.append("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0)
                    sb.append(",");
                sb.append(arr[i]);
            }
            sb.append("]");
        } else if (value instanceof long[]) {
            long[] arr = (long[]) value;
            sb.append("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0)
                    sb.append(",");
                sb.append(arr[i]);
            }
            sb.append("]");
        } else if (value instanceof float[]) {
            float[] arr = (float[]) value;
            sb.append("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0)
                    sb.append(",");
                sb.append(arr[i]);
            }
            sb.append("]");
        } else if (value instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) value;
            sb.append("{");
            boolean first = true;
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                if (!first) {
                    sb.append(",");
                }
                first = false;
                sb.append("\"").append(escapeJson(entry.getKey())).append("\":");
                serializeJsonValue(sb, entry.getValue());
            }
            sb.append("}");
        } else {
            throw new IOException("Unsupported JSON value type: " + value.getClass());
        }
    }

    /**
     * Escape JSON string.
     *
     * @param s input string
     * @return escaped string
     */
    private String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
