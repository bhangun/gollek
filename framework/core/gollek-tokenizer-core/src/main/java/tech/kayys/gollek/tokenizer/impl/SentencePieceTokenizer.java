package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.nativeffi.SentencePieceNative;
import tech.kayys.gollek.tokenizer.spi.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

public class SentencePieceTokenizer implements Tokenizer {

    private final SentencePieceNative nativeLib;
    private final MemorySegment handle;

    public SentencePieceTokenizer(Path libPath, Path modelPath) {
        try (Arena arena = Arena.ofConfined()) {
            this.nativeLib = new SentencePieceNative(libPath.toString());
            this.handle = nativeLib.create();
            nativeLib.load(handle, modelPath.toString(), arena);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        try (Arena arena = Arena.ofConfined()) {
            int[] ids = nativeLib.encode(handle, text, arena);

            long[] result = new long[ids.length];
            for (int i = 0; i < ids.length; i++) {
                result[i] = ids[i];
            }
            return result;

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        try (Arena arena = Arena.ofConfined()) {
            int[] ids = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                ids[i] = (int) tokens[i];
            }
            return nativeLib.decode(handle, ids, arena);

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int vocabSize() {
        return -1;
    } // can extend later

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