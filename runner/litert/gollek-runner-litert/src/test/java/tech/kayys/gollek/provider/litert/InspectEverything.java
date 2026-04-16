package tech.kayys.gollek.provider.litert;
 
 import java.nio.ByteBuffer;
 import java.nio.channels.FileChannel;
 import java.nio.file.*;
 import java.util.ArrayList;
 import java.util.List;
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 
 public class InspectEverything {
     public static void main(String[] args) throws Exception {
         String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm";
         Path p = Paths.get(pathStr);
         long totalSize = Files.size(p);
         
         List<Long> offsets = new ArrayList<>();
         try (FileChannel channel = FileChannel.open(p, StandardOpenOption.READ)) {
             byte[] chunk = new byte[1024 * 1024];
             ByteBuffer buf = ByteBuffer.wrap(chunk);
             for (long pos = 0; pos < totalSize - 16; ) {
                 buf.clear();
                 int read = channel.read(buf, pos);
                 if (read < 16) break;
                 for (int i = 0; i < read - 8; i++) {
                     if (chunk[i+4] == 'T' && chunk[i+5] == 'F' && chunk[i+6] == 'L' && chunk[i+7] == '3') {
                         offsets.add(pos + i);
                     }
                 }
                 pos += (read - 8);
             }
         }
 
         System.out.println("Found " + offsets.size() + " segments.");
         String libraryPath = System.getProperty("LITERT_LIBRARY_PATH");
         if (libraryPath == null) {
             libraryPath = "/Users/bhangun/.gollek/libs/libLiteRt.dylib";
         }
         LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libraryPath));
         
         try (Arena arena = Arena.ofConfined()) {
             MemorySegment env = bindings.createEnvironment(arena);
 
             for (int i = 0; i < offsets.size(); i++) {
                 long off = offsets.get(i);
                 long end = (i + 1 < offsets.size()) ? offsets.get(i + 1) : totalSize;
                 long size = end - off;
                 System.out.println("\n--- Segment " + i + " at 0x" + Long.toHexString(off) + " (" + size + " bytes) ---");
                 
                 try {
                     FileChannel channel = FileChannel.open(p);
                     java.nio.MappedByteBuffer mbb = channel.map(FileChannel.MapMode.READ_ONLY, off, size);
                     MemorySegment segment = MemorySegment.ofBuffer(mbb);
                     
                     MemorySegment model = bindings.createModelFromBuffer(segment, size, arena);
                     MemorySegment opts = bindings.createOptions(arena);
                     
                     // Add Metal Accelerator on Mac
                     if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                         int accel = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu | LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
                         bindings.setOptionsHardwareAccelerators(opts, accel);
                     }
 
                     MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
                     System.out.println("  ✓ CompiledModel created!");
                     
                     int sigs = bindings.getNumModelSignatures(model, arena);
                     System.out.println("  Signatures: " + sigs);
                     for (int s = 0; s < sigs; s++) {
                         MemorySegment sig = bindings.getModelSignature(model, s, arena);
                         String key = bindings.getSignatureKey(sig, arena);
                         System.out.println("    - " + key);
                         
                         int inputs = bindings.getNumSignatureInputs(sig, arena);
                         int outputs = bindings.getNumSignatureOutputs(sig, arena);
                         System.out.println("      Main Subgraph: " + inputs + " In, " + outputs + " Out");
                         for (int inidx = 0; inidx < inputs; inidx++) {
                             System.out.println("        In[" + inidx + "]: " + bindings.getSignatureInputName(sig, inidx, arena));
                         }
                     }
                     
                     bindings.destroyCompiledModel(compiledModel);
                     bindings.destroyOptions(opts);
                     bindings.destroyModel(model);
                 } catch (Exception e) {
                     System.out.println("  ERROR: " + e.getMessage());
                 }
             }
             bindings.destroyEnvironment(env);
         }
     }
 }
