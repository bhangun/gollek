package tech.kayys.gollek.tokenizer.spi;

public class DecodeOptions {
    public boolean skipSpecialTokens = true;

    public static DecodeOptions defaultOptions() {
        return new DecodeOptions();
    }

    public static DecodeOptions builder() {
        return new DecodeOptions();
    }

    public DecodeOptions skipSpecialTokens(boolean skip) {
        this.skipSpecialTokens = skip;
        return this;
    }

    public DecodeOptions build() {
        return this;
    }
}
