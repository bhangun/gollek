
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import java.nio.file.Path;
import java.util.Arrays;

public class TokenizerTest {
    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(args[0]);
        Tokenizer tokenizer = TokenizerFactory.load(modelPath, null);
        String prompt = "who are you";
        long[] ids = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
        System.out.println("Prompt: " + prompt);
        System.out.println("IDs: " + Arrays.toString(ids));
        
        for (long id : ids) {
            System.out.println("ID " + id + " -> [" + tokenizer.decode(new long[]{id}, null) + "]");
        }
    }
}
