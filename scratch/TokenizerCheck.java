import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TokenizerCheck {
    public static void main(String[] args) throws Exception {
        Path modelPath = Paths.get(System.getProperty("user.home"), ".gollek/models/blobs/da96b7da-9cbb-4a74-9694-e597342f2439");
        Tokenizer tokenizer = TokenizerFactory.load(modelPath, null);
        System.out.println("Vocab size: " + tokenizer.vocabSize());
        System.out.println("Token 0: " + tokenizer.decode(new long[]{0}, null));
        System.out.println("Token 100: " + tokenizer.decode(new long[]{100}, null));
        System.out.println("Token 151643 (BOS?): [" + tokenizer.decode(new long[]{151643}, null) + "]");
    }
}
