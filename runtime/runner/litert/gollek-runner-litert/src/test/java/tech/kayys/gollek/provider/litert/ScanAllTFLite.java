package tech.kayys.gollek.provider.litert;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

public class ScanAllTFLite {
    public static void main(String[] args) throws Exception {
        String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm";
        Path path = Paths.get(pathStr);
        
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            System.out.println("Scanning file: " + pathStr + " (" + fileSize + " bytes)");
            
            byte[] chunk = new byte[10 * 1024 * 1024]; // 10MB chunks
            ByteBuffer buf = ByteBuffer.wrap(chunk);
            
            for (long pos = 0; pos < fileSize - 16; ) {
                buf.clear();
                int read = channel.read(buf, pos);
                if (read < 16) break;
                
                for (int i = 0; i < read - 8; i++) {
                    if (chunk[i+4] == 'T' && chunk[i+5] == 'F' && chunk[i+6] == 'L' && chunk[i+7] == '3') {
                        long absolutePos = pos + i;
                        System.out.println("Found TFL3 at 0x" + Long.toHexString(absolutePos) + " (offset " + absolutePos + ")");
                        // Try to read next 32 bytes for context
                        System.out.print("  Magic Context: ");
                        for (int k=0; k<12; k++) {
                            System.out.printf("%02x ", chunk[i+k]);
                        }
                        System.out.println();
                        i += 8; // skip past magic
                    }
                }
                pos += (read - 8);
            }
        }
    }
}
