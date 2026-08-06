package tech.kayys.gollek.onnx.runner;

record OnnxInputTensorCacheStats(
        int inputIdsHits,
        int inputIdsMisses,
        int positionIdsHits,
        int positionIdsMisses,
        int attentionMaskHits,
        int attentionMaskMisses,
        int prefixInputIdsHits,
        int prefixInputIdsMisses,
        int rangePositionIdsHits,
        int rangePositionIdsMisses,
        int evictions) {

    OnnxInputTensorCacheStats(
            int inputIdsHits,
            int inputIdsMisses,
            int positionIdsHits,
            int positionIdsMisses,
            int attentionMaskHits,
            int attentionMaskMisses,
            int prefixInputIdsHits,
            int prefixInputIdsMisses,
            int rangePositionIdsHits,
            int rangePositionIdsMisses) {
        this(inputIdsHits, inputIdsMisses, positionIdsHits, positionIdsMisses, attentionMaskHits,
                attentionMaskMisses, prefixInputIdsHits, prefixInputIdsMisses, rangePositionIdsHits,
                rangePositionIdsMisses, 0);
    }

    OnnxInputTensorCacheStats(
            int inputIdsHits,
            int inputIdsMisses,
            int positionIdsHits,
            int positionIdsMisses,
            int attentionMaskHits,
            int attentionMaskMisses,
            int prefixInputIdsHits,
            int prefixInputIdsMisses) {
        this(inputIdsHits, inputIdsMisses, positionIdsHits, positionIdsMisses, attentionMaskHits,
                attentionMaskMisses, prefixInputIdsHits, prefixInputIdsMisses, 0, 0, 0);
    }

    OnnxInputTensorCacheStats(
            int inputIdsHits,
            int inputIdsMisses,
            int positionIdsHits,
            int positionIdsMisses,
            int attentionMaskHits,
            int attentionMaskMisses) {
        this(inputIdsHits, inputIdsMisses, positionIdsHits, positionIdsMisses, attentionMaskHits,
                attentionMaskMisses, 0, 0, 0, 0, 0);
    }

    OnnxInputTensorCacheStats(
            int inputIdsHits,
            int inputIdsMisses,
            int positionIdsHits,
            int positionIdsMisses) {
        this(inputIdsHits, inputIdsMisses, positionIdsHits, positionIdsMisses, 0, 0, 0, 0, 0, 0, 0);
    }

    int totalHits() {
        return scalarHits() + attentionMaskHits + prefixInputIdsHits + rangePositionIdsHits;
    }

    int totalMisses() {
        return scalarMisses() + attentionMaskMisses + prefixInputIdsMisses + rangePositionIdsMisses;
    }

    int scalarHits() {
        return inputIdsHits + positionIdsHits;
    }

    int scalarMisses() {
        return inputIdsMisses + positionIdsMisses;
    }
}
