/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafeTensorQuantizedReader.java
 * ───────────────────────
 * SafeTensor reader for quantized models.
 */
package tech.kayys.gollek.safetensor.quantization;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Reader for quantized SafeTensor models.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class SafeTensorQuantizedReader {

    private static final Logger log = Logger.getLogger(SafeTensorQuantizedReader.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Read quantized model from SafeTensors format.
     *
     * @param modelPath path to the model file
     * @return map of tensor name to quantized data and metadata
     * @throws IOException if read fails
     */
    public QuantizedModel read(Path modelPath) throws IOException {
        log.infof("Reading quantized SafeTensors model from: %s", modelPath);

        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            // Read header size (8 bytes, little-endian)
            ByteBuffer sizeBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(sizeBuffer);
            sizeBuffer.flip();
            long headerSize = sizeBuffer.getLong();

            log.debugf("Header size: %d bytes", headerSize);

            // Read header
            ByteBuffer headerBuffer = ByteBuffer.allocate((int) headerSize);
            channel.read(headerBuffer);
            headerBuffer.flip();

            byte[] headerBytes = new byte[headerBuffer.remaining()];
            headerBuffer.get(headerBytes);

            String headerJson = new String(headerBytes, java.nio.charset.StandardCharsets.UTF_8);
            log.debugf("Header JSON: %s", headerJson.substring(0, Math.min(200, headerJson.length())));

            // Parse header
            JsonNode header = MAPPER.readTree(headerBytes);

            // Read tensors
            Map<String, byte[]> tensorData = new LinkedHashMap<>();
            Map<String, QuantizedTensorInfo> tensorInfos = new LinkedHashMap<>();
            QuantConfig config = null;

            JsonNode metadata = header.get("__metadata__");
            if (metadata != null && metadata.has("quantization_config")) {
                config = parseQuantizationConfig(metadata.get("quantization_config"));
            }

            Iterator<Map.Entry<String, JsonNode>> fields = header.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                String name = entry.getKey();

                if (name.equals("__metadata__")) {
                    continue;
                }

                JsonNode tensorSpec = entry.getValue();

                // Parse tensor spec
                int[] shape = parseShape(tensorSpec.get("shape"));
                String dtype = tensorSpec.get("dtype").asText();
                JsonNode offsets = tensorSpec.get("data_offsets");
                long startOffset = offsets.get(0).asLong();
                long endOffset = offsets.get(1).asLong();
                long dataSize = endOffset - startOffset;

                // Read tensor data
                byte[] data = new byte[(int) dataSize];
                ByteBuffer dataBuffer = ByteBuffer.wrap(data);

                // Seek to data offset (header size + 8 bytes for header size)
                long dataPosition = 8 + headerSize + startOffset;
                channel.read(dataBuffer, dataPosition);

                tensorData.put(name, data);

                // Parse quantization metadata
                QuantizedTensorInfo info = null;
                if (tensorSpec.has("quantization")) {
                    info = parseQuantizedTensorInfo(name, shape, dtype, tensorSpec.get("quantization"));
                } else {
                    info = QuantizedTensorInfo.builder()
                            .name(name)
                            .shape(shape)
                            .quantizedDtype(dtype)
                            .build();
                }

                tensorInfos.put(name, info);
            }

            QuantizedModel model = new QuantizedModel(tensorData, tensorInfos, config);
            log.infof("Successfully read %d tensors from quantized model", tensorData.size());
            return model;
        }
    }

    /**
     * Read specific tensor from quantized model.
     *
     * @param modelPath  path to the model file
     * @param tensorName name of tensor to read
     * @return tensor data and metadata
     * @throws IOException if read fails
     */
    public QuantizedTensor readTensor(Path modelPath, String tensorName) throws IOException {
        QuantizedModel model = read(modelPath);
        byte[] data = model.data().get(tensorName);
        QuantizedTensorInfo info = model.tensorInfos().get(tensorName);

        if (data == null || info == null) {
            throw new IOException("TorchTensor not found: " + tensorName);
        }

        return new QuantizedTensor(data, info);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Parse quantization config from JSON.
     *
     * @param node JSON node
     * @return quantization configuration
     */
    private QuantConfig parseQuantizationConfig(JsonNode node) {
        QuantConfig.Builder builder = QuantConfig.builder();

        if (node.has("strategy")) {
            try {
                QuantizationEngine.QuantStrategy strategy =
                        QuantizationEngine.QuantStrategy.valueOf(node.get("strategy").asText());
                builder.strategy(strategy);
            } catch (IllegalArgumentException e) {
                log.warnf("Unknown quantization strategy: %s", node.get("strategy").asText());
            }
        }

        if (node.has("bits")) {
            builder.bits(node.get("bits").asInt());
        }
        if (node.has("group_size")) {
            builder.groupSize(node.get("group_size").asInt());
        }
        if (node.has("symmetric")) {
            builder.symmetric(node.get("symmetric").asBoolean());
        }
        if (node.has("per_channel")) {
            builder.perChannel(node.get("per_channel").asBoolean());
        }
        if (node.has("act_order")) {
            builder.actOrder(node.get("act_order").asBoolean());
        }
        if (node.has("damp_percent")) {
            builder.dampPercent(node.get("damp_percent").asDouble());
        }

        return builder.build();
    }

    /**
     * Parse quantized tensor info from JSON.
     *
     * @param name   tensor name
     * @param shape  tensor shape
     * @param dtype  tensor dtype
     * @param node   quantization JSON node
     * @return tensor info
     */
    private QuantizedTensorInfo parseQuantizedTensorInfo(String name, int[] shape, String dtype, JsonNode node) {
        QuantizedTensorInfo.Builder builder = QuantizedTensorInfo.builder()
                .name(name)
                .shape(shape)
                .quantizedDtype(dtype);

        if (node.has("strategy")) {
            try {
                QuantizationEngine.QuantStrategy strategy =
                        QuantizationEngine.QuantStrategy.valueOf(node.get("strategy").asText());
                builder.strategy(strategy);
            } catch (IllegalArgumentException e) {
                log.warnf("Unknown quantization strategy: %s", node.get("strategy").asText());
            }
        }

        if (node.has("group_size")) {
            builder.groupSize(node.get("group_size").asInt());
        }
        if (node.has("channel_axis")) {
            builder.channelAxis(node.get("channel_axis").asInt());
        }
        if (node.has("scales")) {
            builder.scales(parseFloatArray(node.get("scales")));
        }
        if (node.has("zeros")) {
            builder.zeros(parseFloatArray(node.get("zeros")));
        }

        return builder.build();
    }

    /**
     * Parse tensor shape from JSON.
     *
     * @param node JSON node
     * @return shape array
     */
    private int[] parseShape(JsonNode node) {
        int[] shape = new int[node.size()];
        int i = 0;
        for (JsonNode dim : node) {
            shape[i++] = dim.asInt();
        }
        return shape;
    }

    /**
     * Parse float array from JSON.
     *
     * @param node JSON node
     * @return float array
     */
    private float[] parseFloatArray(JsonNode node) {
        float[] arr = new float[node.size()];
        int i = 0;
        for (JsonNode elem : node) {
            arr[i++] = (float) elem.asDouble();
        }
        return arr;
    }

    /**
     * Quantized model data.
     */
    public record QuantizedModel(
            Map<String, byte[]> data,
            Map<String, QuantizedTensorInfo> tensorInfos,
            QuantConfig config) {
    }

    /**
     * Single quantized tensor.
     */
    public record QuantizedTensor(
            byte[] data,
            QuantizedTensorInfo info) {
    }
}
