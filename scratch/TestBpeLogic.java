import java.nio.charset.StandardCharsets;
import java.util.*;

public class TestBpeLogic {
    public static void main(String[] args) {
        Map<Character, String> byteEncoder = buildByteEncoder();
        Map<String, Byte> byteDecoder = new HashMap<>();
        for (Map.Entry<Character, String> e : byteEncoder.entrySet()) {
            byteDecoder.put(e.getValue(), (byte) (int) e.getKey());
        }

        // Test token "Ġis" (found in Qwen vocab)
        String tok = "\u0120is";
        System.out.println("Input token: '" + tok + "'");
        
        String decoded = decodeBytes(tok, byteDecoder);
        System.out.println("Decoded: '" + decoded + "' (length: " + decoded.length() + ")");
        System.out.println("Hex decoded: " + toHex(decoded));
    }

    private static String decodeBytes(String text, Map<String, Byte> byteDecoder) {
        if (text == null || text.isEmpty()) return "";
        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            bytes[i] = byteDecoder.getOrDefault(String.valueOf(c), (byte) c);
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static Map<Character, String> buildByteEncoder() {
        Map<Character, String> map = new HashMap<>();
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) bs.add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.add(i);

        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                map.put((char) b, String.valueOf((char) (256 + n++)));
            } else {
                map.put((char) b, String.valueOf((char) b));
            }
        }
        return map;
    }

    private static String toHex(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            sb.append(String.format("\\u%04x", (int) c));
        }
        return sb.toString();
    }
}
