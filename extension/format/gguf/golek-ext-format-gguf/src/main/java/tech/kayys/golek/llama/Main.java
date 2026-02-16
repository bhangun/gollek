package tech.kayys.golek.llama;

public class Main {
    /*
     * public static void main(String[] args) {
     * try (Arena arena = Arena.ofConfined()) {
     * MemorySegment sysInfo = llama_print_system_info();
     * System.out.println("System info: " + sysInfo.getUtf8String(0));
     * }
     * }
     */

    public static void main(String[] args) throws Throwable {
        // Call ggml_cpu_has_avx
        int hasAvx = (int) llama_h.ggml_cpu_has_avx.HANDLE.invokeExact();

        System.out.println("CPU has AVX: " + (hasAvx != 0));
    }
}
