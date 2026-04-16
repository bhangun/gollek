package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.Test;
import java.nio.file.Path;

public class CompareModelsTest {
    private static int readUint32(byte[] data, int offset) {
        return (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8) | ((data[offset + 2] & 0xFF) << 16) | ((data[offset + 3] & 0xFF) << 24);
    }
    private static int readInt32(byte[] data, int offset) { return readUint32(data, offset); }
    private static int readUint16(byte[] data, int offset) { return (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8); }

    @Test
    public void dumpGeneric() throws Exception {
        Path modelPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "litert", "litert-community", "gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it.litertlm");
        byte[] data = tech.kayys.gollek.provider.litert.LiteRTContainerParser.extractTfliteModel(modelPath);
        
        int rootOffset = readUint32(data, 0);
        int vtableRel = readInt32(data, rootOffset);
        int vtablePos = rootOffset - vtableRel;
        int field4Offset = readUint16(data, vtablePos + 4 + 4 * 2);

        int abs4 = rootOffset + field4Offset;
        int bufVecRel = readUint32(data, abs4);
        int bufVecPos = abs4 + bufVecRel;
        int bufCount = readUint32(data, bufVecPos);

        System.out.println("GENERIC MODEL BUFFERS: " + bufCount);
        for (int i = 0; i < bufCount; i++) {
            int bufEntryRel = readUint32(data, bufVecPos + 4 + i * 4);
            int bufEntryPos = bufVecPos + 4 + i * 4 + bufEntryRel;

            int bvRel = readInt32(data, bufEntryPos);
            int bvPos = bufEntryPos - bvRel;
            int bvSize = readUint16(data, bvPos);
            int bFields = (bvSize - 4) / 2;

            int dataOff = bFields > 0 ? readUint16(data, bvPos + 4) : 0;
            if (dataOff > 0) {
                int dataAbs = bufEntryPos + dataOff;
                int dataRel = readUint32(data, dataAbs);
                int dataPos = dataAbs + dataRel;
                int dataLen = readUint32(data, dataPos);
                if (dataLen > 10000) {
                    System.out.printf("GENERIC BUFFER %d -> size %d\n", i, dataLen);
                }
            }
        }
    }
}
