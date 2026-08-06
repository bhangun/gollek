package tech.kayys.gollek.tokenizer.nativeffi;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

public class SentencePieceNative {

    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup lookup;

    private final MethodHandle spm_create;
    private final MethodHandle spm_destroy;
    private final MethodHandle spm_load;
    private final MethodHandle spm_encode_into;
    private final MethodHandle spm_decode;
    private final MethodHandle spm_free_string;

    public SentencePieceNative(String libPath) {
        System.load(libPath);
        lookup = SymbolLookup.loaderLookup();

        spm_create = downcall("spm_create",
                FunctionDescriptor.of(ADDRESS));

        spm_destroy = downcall("spm_destroy",
                FunctionDescriptor.ofVoid(ADDRESS));

        spm_load = downcall("spm_load",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

        spm_encode_into = downcall("spm_encode_into",
                FunctionDescriptor.of(JAVA_INT,
                        ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));

        spm_decode = downcall("spm_decode",
                FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT));

        spm_free_string = downcall("spm_free_string",
                FunctionDescriptor.ofVoid(ADDRESS));
    }

    private MethodHandle downcall(String name, FunctionDescriptor fd) {
        return linker.downcallHandle(
                lookup.find(name).orElseThrow(),
                fd);
    }

    // =============================

    public MemorySegment create() throws Throwable {
        return (MemorySegment) spm_create.invoke();
    }

    public void destroy(MemorySegment handle) throws Throwable {
        spm_destroy.invoke(handle);
    }

    public void load(MemorySegment handle, String model, Arena arena) throws Throwable {
        MemorySegment cStr = arena.allocateFrom(model);
        int rc = (int) spm_load.invoke(handle, cStr);
        if (rc != 0)
            throw new RuntimeException("Failed to load model");
    }

    public int encodeInto(
            MemorySegment handle,
            String text,
            MemorySegment outBuffer,
            int capacity,
            Arena arena) throws Throwable {

        MemorySegment cText = arena.allocateFrom(text);
        MemorySegment outLen = arena.allocate(JAVA_INT);

        spm_encode_into.invoke(handle, cText, outBuffer, capacity, outLen);

        return outLen.get(JAVA_INT, 0);
    }

    public int[] encode(MemorySegment handle, String text, Arena arena) throws Throwable {
        int tempCapacity = text.length() * 2 + 10;
        MemorySegment tempBuffer = arena.allocate(JAVA_INT, tempCapacity);
        int len = encodeInto(handle, text, tempBuffer, tempCapacity, arena);
        int[] result = new int[len];
        for (int i = 0; i < len; i++) {
            result[i] = tempBuffer.getAtIndex(JAVA_INT, i);
        }
        return result;
    }

    public String decode(MemorySegment handle, int[] tokens, Arena arena) throws Throwable {
        MemorySegment cArray = arena.allocateFrom(JAVA_INT, tokens);

        MemorySegment resultPtr = (MemorySegment) spm_decode.invoke(handle, cArray, tokens.length);

        String result = resultPtr.reinterpret(Long.MAX_VALUE).getString(0);

        spm_free_string.invoke(resultPtr);

        return result;
    }
}
