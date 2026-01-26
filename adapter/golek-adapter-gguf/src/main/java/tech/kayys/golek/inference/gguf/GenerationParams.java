import java.time.Duration;

/**
* Generation parameters for text generation
*/
public class GenerationParams {
private final int maxTokens;
private final float temperature;
private final float topP;
private final int topK;
private final float repeatPenalty;
private final int repeatLastN;
private final boolean stream;

private GenerationParams(Builder builder) {
this.maxTokens = builder.maxTokens;
this.temperature = builder.temperature;
this.topP = builder.topP;
this.topK = builder.topK;
this.repeatPenalty = builder.repeatPenalty;
this.repeatLastN = builder.repeatLastN;
this.stream = builder.stream;
}

public static Builder builder() {
return new Builder();
}

public int getMaxTokens() { return maxTokens; }
public float getTemperature() { return temperature; }
public float getTopP() { return topP; }
public int getTopK() { return topK; }
public float getRepeatPenalty() { return repeatPenalty; }
public int getRepeatLastN() { return repeatLastN; }
public boolean isStream() { return stream; }

public static class Builder {
private int maxTokens = 512;
private float temperature = 0.8f;
private float topP = 0.95f;
private int topK = 40;
private float repeatPenalty = 1.1f;
private int repeatLastN = 64;
private boolean stream = false;

public Builder maxTokens(int maxTokens) {
this.maxTokens = maxTokens;
return this;
}

public Builder temperature(float temperature) {
this.temperature = temperature;
return this;
}

public Builder topP(float topP) {
this.topP = topP;
return this;
}

public Builder topK(int topK) {
this.topK = topK;
return this;
}

public Builder repeatPenalty(float repeatPenalty) {
this.repeatPenalty = repeatPenalty;
return this;
}

public Builder repeatLastN(int repeatLastN) {
this.repeatLastN = repeatLastN;
return this;
}

public Builder timeout(Duration timeout) {
this.timeout = timeout;
return this;
}

public GGUFConfig build() {
return new GGUFConfig(this);
}
}
}