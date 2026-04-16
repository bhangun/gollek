/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorFFMLoaderTest.java
 * ────────────────────────────
 * Comprehensive unit and integration tests for the SafeTensors FFM loader.
 *
 * Test strategy
 * ═════════════
 *   1. In-memory generation of valid SafeTensors binary blobs to avoid
 *      external file dependencies.
 *   2. Structural tests: header length prefix, JSON parsing, tensor metadata.
 *   3. TorchTensor access tests: dtype, shape, element values, slicing.
 *   4. Error path tests: corrupt headers, truncated files, bad offsets.
 *   5. F16/BF16 conversion tests.
 *   6. Multi-shard index parsing tests.
 *
 * All tests are JUnit 5 + AssertJ.  No Quarkus test harness required for
 * unit tests (the tests instantiate components directly).
 * Integration tests (SafetensorQuarkusIT) use @QuarkusTest.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import tech.kayys.gollek.safetensor.exception.SafetensorException;

import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for the SafeTensors FFM loader stack.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SafetensorFFMLoaderTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static SafetensorHeaderParser PARSER;

    @TempDir
    Path tempDir;

    @BeforeAll
    static void setupParser() {
        PARSER = SafetensorHeaderParser.create(MAPPER);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helper: build a minimal SafeTensors binary blob in memory
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Build a valid SafeTensors file in a byte array.
     *
     * @param json      the header JSON string
     * @param dataBytes the tensor data bytes
     * @return complete SafeTensors file bytes
     */
    private static byte[] buildSafetensors(String json, byte[] dataBytes) {
        byte[] jsonBytes = json.getBytes(StandardCharsets.UTF_8);
        // 8-byte LE uint64 header length
        ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        lenBuf.putLong(jsonBytes.length);
        byte[] lenBytes = lenBuf.array();

        byte[] result = new byte[8 + jsonBytes.length + (dataBytes != null ? dataBytes.length : 0)];
        System.arraycopy(lenBytes, 0, result, 0, 8);
        System.arraycopy(jsonBytes, 0, result, 8, jsonBytes.length);
        if (dataBytes != null) {
            System.arraycopy(dataBytes, 0, result, 8 + jsonBytes.length, dataBytes.length);
        }
        return result;
    }

    /**
     * Wrap a byte array in a MemorySegment using a confined arena (test-scoped).
     */
    private static MemorySegment toSegment(byte[] bytes, Arena arena) {
        MemorySegment seg = arena.allocate(bytes.length);
        MemorySegment.copy(bytes, 0, seg, ValueLayout.JAVA_BYTE, 0, bytes.length);
        return seg;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. SafetensorDType tests
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    void dtype_fromJson_knownTypes() {
        assertThat(SafetensorDType.fromJson("F32")).isEqualTo(SafetensorDType.F32);
        assertThat(SafetensorDType.fromJson("BF16")).isEqualTo(SafetensorDType.BF16);
        assertThat(SafetensorDType.fromJson("F16")).isEqualTo(SafetensorDType.F16);
        assertThat(SafetensorDType.fromJson("I8")).isEqualTo(SafetensorDType.I8);
        assertThat(SafetensorDType.fromJson("BOOL")).isEqualTo(SafetensorDType.BOOL);
    }

    @Test
    @Order(2)
    void dtype_fromJson_caseInsensitive() {
        assertThat(SafetensorDType.fromJson("f32")).isEqualTo(SafetensorDType.F32);
        assertThat(SafetensorDType.fromJson("bf16")).isEqualTo(SafetensorDType.BF16);
    }

    @Test
    @Order(3)
    void dtype_fromJson_unknownThrows() {
        assertThatThrownBy(() -> SafetensorDType.fromJson("FLOAT32"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unknown SafeTensors dtype");
    }

    @ParameterizedTest
    @Order(4)
    @EnumSource(SafetensorDType.class)
    void dtype_byteSizeIsPositive(SafetensorDType dtype) {
        assertThat(dtype.byteSize()).isGreaterThan(0);
    }

    @Test
    @Order(5)
    void dtype_totalBytes_f32_2d() {
        assertThat(SafetensorDType.F32.totalBytes(4, 8)).isEqualTo(128L);
    }

    @Test
    @Order(6)
    void dtype_totalBytes_overflow_returnsMinusOne() {
        // 2^31 × 2^31 would overflow a long
        assertThat(SafetensorDType.F32.totalBytes(Long.MAX_VALUE / 4 + 1, 8))
                .isEqualTo(-1L);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. SafetensorTensorInfo tests
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(10)
    void tensorInfo_basicConstruction() {
        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "weight", SafetensorDType.F32, new long[] { 2, 3 }, 0L, 24L);

        assertThat(info.name()).isEqualTo("weight");
        assertThat(info.dtype()).isEqualTo(SafetensorDType.F32);
        assertThat(info.shape()).isEqualTo(new long[] { 2, 3 });
        assertThat(info.byteLength()).isEqualTo(24L);
        assertThat(info.numElements()).isEqualTo(6L);
        assertThat(info.rank()).isEqualTo(2);
    }

    @Test
    @Order(11)
    void tensorInfo_byteLengthMismatch_throws() {
        assertThatThrownBy(() -> SafetensorTensorInfo.of("w", SafetensorDType.F32, new long[] { 2, 2 }, 0L, 20L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Byte-length mismatch");
    }

    @Test
    @Order(12)
    void tensorInfo_elementByteOffset_rowMajor() {
        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "m", SafetensorDType.F32, new long[] { 3, 4 }, 0L, 48L);

        // Row-major: element [1][2] = 1*4 + 2 = 6 elements → 24 bytes
        assertThat(info.elementByteOffset(1L, 2L)).isEqualTo(24L);
    }

    @Test
    @Order(13)
    void tensorInfo_scalar_numElementsOne() {
        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "s", SafetensorDType.F32, new long[] {}, 0L, 4L);
        assertThat(info.isScalar()).isTrue();
        assertThat(info.numElements()).isEqualTo(1L);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. SafetensorHeaderParser tests
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(20)
    void parser_minimalValidFile() {
        // A file with one F32 tensor: shape [2,2], data at [0,16)
        String json = """
                {
                  "w": {"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}
                }""";
        byte[] data = new float[] { 1f, 2f, 3f, 4f }.toString().getBytes(); // dummy
        byte[] payload = buildSafetensors(json, new byte[16]);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = toSegment(payload, arena);
            SafetensorHeader header = PARSER.parse(seg, Path.of("test.safetensors"));

            assertThat(header.tensorCount()).isEqualTo(1);
            assertThat(header.hasTensor("w")).isTrue();

            SafetensorTensorInfo info = header.tensor("w");
            assertThat(info.dtype()).isEqualTo(SafetensorDType.F32);
            assertThat(info.shape()).isEqualTo(new long[] { 2, 2 });
            assertThat(info.byteLength()).isEqualTo(16L);
        }
    }

    @Test
    @Order(21)
    void parser_withMetadata() {
        String json = """
                {
                  "__metadata__": {"format":"pt","version":"2.0"},
                  "bias": {"dtype":"F32","shape":[4],"data_offsets":[0,16]}
                }""";
        byte[] payload = buildSafetensors(json, new byte[16]);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = toSegment(payload, arena);
            SafetensorHeader header = PARSER.parse(seg, Path.of("test.safetensors"));

            assertThat(header.fileMetadata()).containsEntry("format", "pt");
            assertThat(header.fileMetadata()).containsEntry("version", "2.0");
        }
    }

    @Test
    @Order(22)
    void parser_truncatedFile_throws() {
        // Only 4 bytes — too short even for the length prefix
        byte[] tiny = new byte[] { 0, 0, 0, 0 };
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = toSegment(tiny, arena);
            assertThatThrownBy(() -> PARSER.parse(seg, Path.of("bad.safetensors")))
                    .isInstanceOf(SafetensorException.ValidationException.class);
        }
    }

    @Test
    @Order(23)
    void parser_overlappingOffsets_throws() {
        // Tensors a and b overlap: a=[0,16), b=[8,24)
        String json = """
                {
                  "a": {"dtype":"F32","shape":[4],"data_offsets":[0,16]},
                  "b": {"dtype":"F32","shape":[4],"data_offsets":[8,24]}
                }""";
        byte[] payload = buildSafetensors(json, new byte[24]);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = toSegment(payload, arena);
            assertThatThrownBy(() -> PARSER.parse(seg, Path.of("overlap.safetensors")))
                    .isInstanceOf(SafetensorException.ValidationException.class)
                    .hasMessageContaining("overlaps");
        }
    }

    @Test
    @Order(24)
    void parser_unknownDtype_throws() {
        String json = """
                {"x": {"dtype":"FLOAT128","shape":[4],"data_offsets":[0,64]}}""";
        byte[] payload = buildSafetensors(json, new byte[64]);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = toSegment(payload, arena);
            assertThatThrownBy(() -> PARSER.parse(seg, Path.of("bad.safetensors")))
                    .isInstanceOf(SafetensorException.HeaderParseException.class);
        }
    }

    @Test
    @Order(25)
    void parser_multiTensor_orderPreserved() {
        String json = """
                {
                  "first":  {"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                  "second": {"dtype":"I8", "shape":[4],"data_offsets":[4,8]},
                  "third":  {"dtype":"F64","shape":[1],"data_offsets":[8,16]}
                }""";
        byte[] payload = buildSafetensors(json, new byte[16]);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = toSegment(payload, arena);
            SafetensorHeader header = PARSER.parse(seg, Path.of("multi.safetensors"));

            // Verify insertion order is preserved
            assertThat(header.tensors().keySet()).containsExactly("first", "second", "third");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. SafetensorTensor element access tests
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(30)
    void tensor_f32_elementAccess() {
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        byte[] data = floatsToBytes(values);

        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "w", SafetensorDType.F32, new long[] { 2, 3 }, 0L, data.length);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(data.length);
            MemorySegment.copy(data, 0, seg, ValueLayout.JAVA_BYTE, 0, data.length);

            SafetensorTensor tensor = new SafetensorTensor(info, seg);

            assertThat(tensor.getFloat(0, 0)).isEqualTo(1.0f);
            assertThat(tensor.getFloat(0, 2)).isEqualTo(3.0f);
            assertThat(tensor.getFloat(1, 2)).isEqualTo(6.0f);
        }
    }

    @Test
    @Order(31)
    void tensor_f32_toFloatArray() {
        float[] expected = { 10.0f, 20.0f, 30.0f };
        byte[] data = floatsToBytes(expected);

        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "v", SafetensorDType.F32, new long[] { 3 }, 0L, data.length);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(data.length);
            MemorySegment.copy(data, 0, seg, ValueLayout.JAVA_BYTE, 0, data.length);

            SafetensorTensor tensor = new SafetensorTensor(info, seg);
            assertThat(tensor.toFloatArray()).isEqualTo(expected);
        }
    }

    @Test
    @Order(32)
    void tensor_f16_toBF16AsFloat() {
        // BF16 representation of 1.0f: take top 16 bits of F32 bit pattern
        float original = 1.0f;
        int f32bits = Float.floatToIntBits(original);
        short bf16bits = (short) (f32bits >>> 16);

        byte[] data = new byte[2];
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putShort(bf16bits);

        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "b", SafetensorDType.BF16, new long[] { 1 }, 0L, 2L);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(2);
            MemorySegment.copy(data, 0, seg, ValueLayout.JAVA_BYTE, 0, 2);

            SafetensorTensor tensor = new SafetensorTensor(info, seg);
            float result = tensor.getBF16AsFloat(0L);
            assertThat(result).isCloseTo(original, within(0.01f));
        }
    }

    @Test
    @Order(33)
    void tensor_closedAccess_throws() {
        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "x", SafetensorDType.F32, new long[] { 1 }, 0L, 4L);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(4);
            SafetensorTensor tensor = new SafetensorTensor(info, seg);
            tensor.close();

            assertThatThrownBy(tensor::segment)
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("closed");
        }
    }

    @Test
    @Order(34)
    void tensor_wrongDtype_throws() {
        SafetensorTensorInfo info = SafetensorTensorInfo.of(
                "i", SafetensorDType.I32, new long[] { 1 }, 0L, 4L);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(4);
            SafetensorTensor tensor = new SafetensorTensor(info, seg);

            // toFloatArray() should fail because dtype is I32, not F32
            assertThatThrownBy(tensor::toFloatArray)
                    .isInstanceOf(UnsupportedOperationException.class)
                    .hasMessageContaining("F32");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. SafetensorLoadResult tests
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(40)
    void loadResult_tensorByName() {
        String json = """
                {"embed": {"dtype":"F32","shape":[2],"data_offsets":[0,8]}}""";
        float[] floats = { 1.5f, 2.5f };
        byte[] payload = buildSafetensors(json, floatsToBytes(floats));

        try (Arena arena = Arena.ofShared()) {
            MemorySegment seg = toSegment(payload, arena);
            SafetensorHeader header = PARSER.parse(seg, Path.of("r.safetensors"));

            SafetensorLoadResult result = new SafetensorLoadResult(
                    Path.of("r.safetensors"), header, seg, arena,
                    SafetensorLoadResult.LoadMode.MMAP);

            SafetensorTensor t = result.tensor("embed");
            assertThat(t.name()).isEqualTo("embed");
            assertThat(t.toFloatArray()).isEqualTo(floats);

            result.close();
        }
    }

    @Test
    @Order(41)
    void loadResult_closedAccess_throws() {
        String json = """
                {"w": {"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""";
        byte[] payload = buildSafetensors(json, new byte[4]);

        Arena arena = Arena.ofShared();
        MemorySegment seg = toSegment(payload, arena);
        SafetensorHeader header = PARSER.parse(seg, Path.of("c.safetensors"));

        SafetensorLoadResult result = new SafetensorLoadResult(
                Path.of("c.safetensors"), header, seg, arena,
                SafetensorLoadResult.LoadMode.COPY);
        result.close();

        assertThatThrownBy(result::header)
                .isInstanceOf(IllegalStateException.class);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. ShardIndex parsing tests
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(50)
    void shardIndex_parse_valid(@TempDir Path dir) throws Exception {
        // Create dummy shard files
        Files.writeString(dir.resolve("model-00001-of-00002.safetensors"), "");
        Files.writeString(dir.resolve("model-00002-of-00002.safetensors"), "");

        String indexJson = """
                {
                  "metadata": {"total_size": 1000000},
                  "weight_map": {
                    "embed.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors"
                  }
                }""";
        Path indexPath = dir.resolve("model.safetensors.index.json");
        Files.writeString(indexPath, indexJson);

        SafetensorShardIndex index = SafetensorShardIndex.load(indexPath, MAPPER);

        assertThat(index.shardCount()).isEqualTo(2);
        assertThat(index.tensorCount()).isEqualTo(2);
        assertThat(index.totalSize()).isEqualTo(1_000_000L);
        assertThat(index.hasTensor("embed.weight")).isTrue();
        assertThat(index.shardFileNames()).containsExactly(
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors");
    }

    @Test
    @Order(51)
    void shardIndex_missingShard_throws(@TempDir Path dir) throws Exception {
        // Only create one of the two referenced shards
        Files.writeString(dir.resolve("model-00001-of-00002.safetensors"), "");
        // model-00002... is intentionally missing

        String indexJson = """
                {
                  "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors"
                  }
                }""";
        Path indexPath = dir.resolve("model.safetensors.index.json");
        Files.writeString(indexPath, indexJson);

        assertThatThrownBy(() -> SafetensorShardIndex.load(indexPath, MAPPER))
                .isInstanceOf(SafetensorException.ShardException.class)
                .hasMessageContaining("missing shard");
    }

    @Test
    @Order(52)
    void shardIndex_isShardedModel_detects(@TempDir Path dir) throws Exception {
        assertThat(SafetensorShardIndex.isShardedModel(dir)).isFalse();

        Files.writeString(dir.resolve("model.safetensors.index.json"), "{}");
        assertThat(SafetensorShardIndex.isShardedModel(dir)).isTrue();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 7. File I/O tests (write real files, load via FFM mmap)
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @Order(60)
    void ffmLoader_loadRealFile_mmap(@TempDir Path dir) throws Exception {
        float[] weights = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }; // 2×4 matrix
        String json = """
                {"layer.weight": {"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}""";
        byte[] payload = buildSafetensors(json, floatsToBytes(weights));

        Path filePath = dir.resolve("model.safetensors");
        Files.write(filePath, payload);

        // Use header parser directly with a file-based loader
        try (Arena arena = Arena.ofShared()) {
            java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(filePath,
                    java.nio.file.StandardOpenOption.READ);
            long size = ch.size();
            MemorySegment seg = ch.map(
                    java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, size, arena);
            ch.close();

            SafetensorHeader header = PARSER.parse(seg, filePath);
            SafetensorLoadResult result = new SafetensorLoadResult(
                    filePath, header, seg, arena, SafetensorLoadResult.LoadMode.MMAP);

            SafetensorTensor tensor = result.tensor("layer.weight");
            assertThat(tensor.shape()).isEqualTo(new long[] { 2, 4 });
            assertThat(tensor.toFloatArray()).isEqualTo(weights);

            result.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static byte[] floatsToBytes(float[] floats) {
        ByteBuffer buf = ByteBuffer.allocate(floats.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : floats)
            buf.putFloat(f);
        return buf.array();
    }
}
