package tech.kayys.gollek.provider.litert;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class ScanTaskSegments {
    public static void main(String[] args) throws Exception {
        String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.task";
        Path path = Paths.get(pathStr);
        
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            System.out.println("Scanning file: " + pathStr + " (" + fileSize + " bytes)");
            
            byte[] chunk = new byte[10 * 1024 * 1024]; 
            ByteBuffer buf = ByteBuffer.wrap(chunk);
            
            for (long pos = 0; pos < fileSize - 16; ) {
                buf.clear();
                int read = channel.read(buf, pos);
                if (read < 16) break;
                
                for (int i = 0; i < read - 8; i++) {
                    if (chunk[i+4] == 'T' && chunk[i+5] == 'F' && chunk[i+6] == 'L' && chunk[i+7] == '3') {
                        long absolutePos = pos + i;
                        System.out.println("Found TFL3 at 0x" + Long.toHexString(absolutePos) + " (offset " + absolutePos + ")");
                        i += 8; 
                    }
                }
                pos += (read - 8);
            }
        }
    }
}
