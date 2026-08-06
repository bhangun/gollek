import tech.kayys.alkhawarizm.gguf.loader.GGUFLoader;
import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;
import tech.kayys.alkhawarizm.gguf.tokenizer.GGUFTokenizer;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

import java.nio.file.Path;
import java.util.Arrays;

public final class GgufTokenizerProbe {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: GgufTokenizerProbe <path-to-model.gguf> <text>");
            System.exit(1);
        }

        Path modelPath = Path.of(args[0]);
        String text = args[1];

        try (GGUFModel model = GGUFLoader.loadModel(modelPath)) {
            GGUFTokenizer tokenizer = new GGUFTokenizer(model);
            long[] ids = tokenizer.encode(text, EncodeOptions.defaultOptions());
            System.out.println("encoded.ids=" + Arrays.toString(ids));
            System.out.println("decoded.roundtrip=" + tokenizer.decode(ids, DecodeOptions.defaultOptions()));
            System.out.println("bos=" + tokenizer.getBosTokenId() + " eos=" + tokenizer.getEosTokenId() + " pad=" + tokenizer.getPadTokenId());
            System.out.println("stop.ids=" + Arrays.toString(tokenizer.allStopTokenIds()));
            System.out.println("specials=" + tokenizer.specialTokens());
        }
    }
}
