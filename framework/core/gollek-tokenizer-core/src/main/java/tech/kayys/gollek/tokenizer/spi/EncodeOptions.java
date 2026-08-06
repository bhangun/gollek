package tech.kayys.gollek.tokenizer.spi;

public class EncodeOptions {
    public boolean addBos = false;
    public boolean addEos = false;

    public static EncodeOptions defaultOptions() {
        return new EncodeOptions();
    }

    public static EncodeOptions builder() {
        return new EncodeOptions();
    }

    public EncodeOptions addBos(boolean addBos) {
        this.addBos = addBos;
        return this;
    }

    public EncodeOptions addEos(boolean addEos) {
        this.addEos = addEos;
        return this;
    }

    public EncodeOptions build() {
        return this;
    }
}