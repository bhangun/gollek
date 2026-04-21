import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.Arena;
public class TestMemoryCopy {
    public static void main(String[] args) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocateArray(ValueLayout.JAVA_FLOAT, 10);
            MemorySegment dst = arena.allocateArray(ValueLayout.JAVA_FLOAT, 10);
            for(int i=0;i<10;i++) src.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)i);
            
            MemorySegment.copy(src, ValueLayout.JAVA_FLOAT, 0L, dst, ValueLayout.JAVA_FLOAT, 0L, 5L);
            System.out.println("dst[4]=" + dst.getAtIndex(ValueLayout.JAVA_FLOAT, 4));
            System.out.println("dst[5]=" + dst.getAtIndex(ValueLayout.JAVA_FLOAT, 5));
        }
    }
}
