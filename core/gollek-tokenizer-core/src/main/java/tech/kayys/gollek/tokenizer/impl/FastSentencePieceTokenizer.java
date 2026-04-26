package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.nativeffi.SentencePieceNative;
import tech.kayys.gollek.tokenizer.spi.*;

import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class FastSentencePieceTokenizer implements Tokenizer, AutoCloseable {

    private final SentencePieceNative nativeLib;

    // ✅ PERSISTENT
    private final MemorySegment handle;
    private final Arena arena;

    // ✅ REUSABLE BUFFER (no malloc)
    private final MemorySegment buffer;
    private final int capacity = 8192;

    public FastSentencePieceTokenizer(Path libPath, Path modelPath) {
        try {
            this.nativeLib = new SentencePieceNative(libPath.toString());

            // shared arena = long-lived
            this.arena = Arena.ofAuto();

            // create ONCE
            this.handle = nativeLib.create();

            // load ONCE
            nativeLib.load(handle, modelPath.toString(), arena);

            // allocate buffer ONCE
            this.buffer = arena.allocate(JAVA_INT, capacity);

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        try {
            int len = nativeLib.encodeInto(
                    handle,
                    text,
                    buffer,
                    capacity,
                    arena);

            List<Long> tokens = new ArrayList<>();
            if (options.addBos) {
                int bos = bosTokenId();
                if (bos != -1) tokens.add((long) bos);
            }

            for (int i = 0; i < len; i++) {
                tokens.add((long) buffer.getAtIndex(JAVA_INT, i));
            }

            if (options.addEos) {
                int eos = eosTokenId();
                if (eos != -1) tokens.add((long) eos);
            }

            return tokens.stream().mapToLong(Long::longValue).toArray();

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        try {
            int[] ids = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                ids[i] = (int) tokens[i];
            }

            return nativeLib.decode(handle, ids, arena);

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    // cleanup
    @Override
    public void close() {
        try {
            nativeLib.destroy(handle);
            arena.close();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int vocabSize() {
        return -1;
    }

    @Override
    public int bosTokenId() {
        return -1;
    }

    @Override
    public int eosTokenId() {
        return -1;
    }

    @Override
    public int padTokenId() {
        return -1;
    }

    @Override
    public int[] allStopTokenIds() {
        int eos = eosTokenId();
        return eos != -1 ? new int[]{eos} : new int[0];
    }
}