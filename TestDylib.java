import java.lang.foreign.*;
import java.lang.invoke.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

public class TestDylib {
    public static void main(String[] args) throws Throwable {
        Path bridge = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libgollek_gguf_fast.dylib").toAbsolutePath();
        System.load(bridge.toString());
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.libraryLookup(bridge, Arena.global());
        
        MethodHandle open = linker.downcallHandle(lookup.find("gollek_gguf_open").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        MethodHandle generate = linker.downcallHandle(lookup.find("gollek_gguf_generate").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        MethodHandle close = linker.downcallHandle(lookup.find("gollek_gguf_close").orElseThrow(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        String modelStr = "/Users/bhangun/.gollek/models/blobs/unsloth/gemma-4-12b-it-GGUF/unsloth__gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf";
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment model = arena.allocateFrom(modelStr, StandardCharsets.UTF_8);
            MemorySegment backendDir = arena.allocateFrom(System.getProperty("user.home") + "/.gollek/libs/llama", StandardCharsets.UTF_8);
            MemorySegment error = arena.allocate(8192);
            
            System.out.println("Opening...");
            MemorySegment handle = (MemorySegment) open.invokeExact(model, backendDir, 512, 512, 512, 8, 8, -1, 1, 0, 0, error, 8192L);
            if (handle.address() == 0L) {
                System.out.println("Error opening: " + error.getString(0));
                return;
            }
            
            MemorySegment prompt = arena.allocateFrom("<bos><start_of_turn>user\nwho are you<end_of_turn>\n<start_of_turn>model\n", StandardCharsets.UTF_8);
            MemorySegment output = arena.allocate(65536);
            
            System.out.println("Generating 1...");
            int flag = 0;
            int generated = (int) generate.invokeExact(handle, prompt, 20, 0.0f, 40, 0.9f, flag, output, 65536L, error, 8192L);
            System.out.println("Generated " + generated + ":\n" + output.getString(0));

            System.out.println("Generating 2...");
            generated = (int) generate.invokeExact(handle, prompt, 20, 0.0f, 40, 0.9f, 0, output, 65536L, error, 8192L);
            System.out.println("Generated " + generated + ":\n" + output.getString(0));
            
            close.invokeExact(handle);
        }
    }
}
