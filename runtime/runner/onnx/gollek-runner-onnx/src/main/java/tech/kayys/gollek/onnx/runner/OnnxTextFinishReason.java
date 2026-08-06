package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.spi.inference.InferenceResponse;

enum OnnxTextFinishReason {
    STOP("stop", InferenceResponse.FinishReason.STOP),
    LENGTH("length", InferenceResponse.FinishReason.LENGTH),
    CANCELLED("cancelled", InferenceResponse.FinishReason.ERROR);

    private final String wireValue;
    private final InferenceResponse.FinishReason responseReason;

    OnnxTextFinishReason(String wireValue, InferenceResponse.FinishReason responseReason) {
        this.wireValue = wireValue;
        this.responseReason = responseReason;
    }

    String wireValue() {
        return wireValue;
    }

    InferenceResponse.FinishReason responseReason() {
        return responseReason;
    }
}
