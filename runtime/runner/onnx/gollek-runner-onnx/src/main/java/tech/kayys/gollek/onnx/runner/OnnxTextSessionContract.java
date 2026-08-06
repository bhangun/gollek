package tech.kayys.gollek.onnx.runner;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

final class OnnxTextSessionContract {

    private static final TensorNames DEFAULT_NAMES = new TensorNames(
            "input_ids",
            "attention_mask",
            "position_ids",
            "logits",
            "past_key_values.%d.key",
            "past_key_values.%d.value",
            "present.%d.key",
            "present.%d.value");

    private final String[] sessionInputNames;
    private final String[] sessionOutputNames;
    private final String[] runInputNames;
    private final String[] pastKvInputNames;
    private final String[] runOutputNames;
    private final TensorNames tensorNames;
    private final boolean hasKvInputs;
    private final boolean hasPositionIds;
    private final int kvLayerCount;

    private OnnxTextSessionContract(
            String[] sessionInputNames,
            String[] sessionOutputNames,
            String[] runInputNames,
            String[] pastKvInputNames,
            String[] runOutputNames,
            TensorNames tensorNames,
            boolean hasKvInputs,
            boolean hasPositionIds,
            int kvLayerCount) {
        this.sessionInputNames = copy(sessionInputNames);
        this.sessionOutputNames = copy(sessionOutputNames);
        this.runInputNames = copy(runInputNames);
        this.pastKvInputNames = copy(pastKvInputNames);
        this.runOutputNames = copy(runOutputNames);
        this.tensorNames = Objects.requireNonNull(tensorNames, "tensorNames");
        this.hasKvInputs = hasKvInputs;
        this.hasPositionIds = hasPositionIds;
        if (kvLayerCount < 0) {
            throw new IllegalArgumentException("kvLayerCount must be >= 0");
        }
        this.kvLayerCount = kvLayerCount;
    }

    static OnnxTextSessionContract empty() {
        return new OnnxTextSessionContract(
                new String[0],
                new String[0],
                new String[] { DEFAULT_NAMES.inputIdsName(), DEFAULT_NAMES.attentionMaskName() },
                new String[0],
                new String[] { DEFAULT_NAMES.logitsName() },
                DEFAULT_NAMES,
                false,
                false,
                0);
    }

    static OnnxTextSessionContract create(
            String[] inputNames,
            String[] outputNames,
            TensorNames tensorNames,
            boolean usePastKvCache,
            int fallbackKvLayers) {
        TensorNames names = Objects.requireNonNull(tensorNames, "tensorNames").withDefaults(DEFAULT_NAMES);
        int configuredKvLayers = Math.max(1, fallbackKvLayers);
        Set<String> availableInputs = stringSet(inputNames);
        Set<String> availableOutputs = stringSet(outputNames);
        boolean hasPositionIds = availableInputs.contains(names.positionIdsName());
        String[] baseInputNames = hasPositionIds
                ? new String[] { names.inputIdsName(), names.attentionMaskName(), names.positionIdsName() }
                : new String[] { names.inputIdsName(), names.attentionMaskName() };

        String[] pastKvNames = buildPastKvInputNames(
                inputNames,
                names.pastKeyNameTemplate(),
                names.pastValueNameTemplate(),
                configuredKvLayers);
        String[] presentNames = buildPresentKvOutputNames(
                inputNames,
                outputNames,
                names.pastKeyNameTemplate(),
                names.presentKeyNameTemplate(),
                names.presentValueNameTemplate(),
                configuredKvLayers);
        boolean hasKvInputs = usePastKvCache
                && pastKvNames.length > 0
                && presentNames.length > 0
                && availableInputs.containsAll(Set.of(pastKvNames))
                && availableOutputs.containsAll(Set.of(presentNames));
        String[] runInputNames = hasKvInputs ? concat(baseInputNames, pastKvNames) : baseInputNames;
        String[] runOutputNames = hasKvInputs
                ? concat(new String[] { names.logitsName() }, presentNames)
                : new String[] { names.logitsName() };
        return new OnnxTextSessionContract(
                inputNames,
                outputNames,
                runInputNames,
                hasKvInputs ? pastKvNames : new String[0],
                runOutputNames,
                names,
                hasKvInputs,
                hasPositionIds,
                hasKvInputs ? pastKvNames.length / 2 : 0);
    }

    String[] sessionInputNames() {
        return sessionInputNames;
    }

    String[] sessionOutputNames() {
        return sessionOutputNames;
    }

    String[] runInputNames() {
        return runInputNames;
    }

    String[] pastKvInputNames() {
        return pastKvInputNames;
    }

    String[] runOutputNames() {
        return runOutputNames;
    }

    String logitsName() {
        return tensorNames.logitsName();
    }

    boolean hasKvInputs() {
        return hasKvInputs;
    }

    boolean hasPositionIds() {
        return hasPositionIds;
    }

    int kvLayerCount() {
        return kvLayerCount;
    }

    int pastKvInputCount() {
        return pastKvInputNames.length;
    }

    private static String[] buildPastKvInputNames(
            String[] inputNames,
            String pastKeyNameTemplate,
            String pastValueNameTemplate,
            int configuredKvLayers) {
        int layers = detectedKvLayerCount(inputNames, pastKeyNameTemplate, "past_key_values.", ".key",
                configuredKvLayers);
        String[] names = new String[layers * 2];
        for (int layer = 0; layer < layers; layer++) {
            names[layer * 2] = indexedName(pastKeyNameTemplate, layer, "past_key_values.%d.key");
            names[layer * 2 + 1] = indexedName(pastValueNameTemplate, layer, "past_key_values.%d.value");
        }
        return names;
    }

    private static String[] buildPresentKvOutputNames(
            String[] inputNames,
            String[] outputNames,
            String pastKeyNameTemplate,
            String presentKeyNameTemplate,
            String presentValueNameTemplate,
            int configuredKvLayers) {
        int layers = detectedKvLayerCount(outputNames, presentKeyNameTemplate, "present.", ".key",
                configuredKvLayers);
        if (layers == 0) {
            layers = detectedKvLayerCount(inputNames, pastKeyNameTemplate, "past_key_values.", ".key",
                    configuredKvLayers);
        }
        String[] names = new String[layers * 2];
        for (int layer = 0; layer < layers; layer++) {
            names[layer * 2] = indexedName(presentKeyNameTemplate, layer, "present.%d.key");
            names[layer * 2 + 1] = indexedName(presentValueNameTemplate, layer, "present.%d.value");
        }
        return names;
    }

    private static int detectedKvLayerCount(
            String[] names,
            String template,
            String legacyPrefix,
            String legacySuffix,
            int configuredKvLayers) {
        if (names == null || names.length == 0) {
            return configuredKvLayers;
        }
        Set<String> available = stringSet(names);
        int max = Math.max(64, configuredKvLayers + 8);
        int detected = 0;
        for (int layer = 0; layer < max; layer++) {
            if (available.contains(indexedName(template, layer, legacyPrefix + "%d" + legacySuffix))) {
                detected = layer + 1;
            }
        }
        if (detected > 0) {
            return detected;
        }
        for (String name : names) {
            if (name != null && name.startsWith(legacyPrefix) && name.endsWith(legacySuffix)) {
                detected++;
            }
        }
        return detected > 0 ? detected : configuredKvLayers;
    }

    private static String indexedName(String template, int index, String fallbackTemplate) {
        String pattern = template == null || template.isBlank() ? fallbackTemplate : template;
        if (pattern.contains("%d")) {
            try {
                return String.format(java.util.Locale.ROOT, pattern, index);
            } catch (Exception ignored) {
                return String.format(java.util.Locale.ROOT, fallbackTemplate, index);
            }
        }
        if (pattern.contains("{}")) {
            return pattern.replace("{}", Integer.toString(index));
        }
        if (pattern.contains("{layer}")) {
            return pattern.replace("{layer}", Integer.toString(index));
        }
        return index == 0 ? pattern : pattern + index;
    }

    private static Set<String> stringSet(String[] values) {
        Set<String> result = new LinkedHashSet<>();
        if (values == null) {
            return result;
        }
        for (String value : values) {
            if (value != null) {
                result.add(value);
            }
        }
        return result;
    }

    private static String[] concat(String[] first, String[] second) {
        String[] result = new String[first.length + second.length];
        System.arraycopy(first, 0, result, 0, first.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }

    private static String[] copy(String[] values) {
        return values == null ? new String[0] : values.clone();
    }

    record TensorNames(
            String inputIdsName,
            String attentionMaskName,
            String positionIdsName,
            String logitsName,
            String pastKeyNameTemplate,
            String pastValueNameTemplate,
            String presentKeyNameTemplate,
            String presentValueNameTemplate) {

        private TensorNames withDefaults(TensorNames defaults) {
            return new TensorNames(
                    nonBlank(inputIdsName, defaults.inputIdsName),
                    nonBlank(attentionMaskName, defaults.attentionMaskName),
                    nonBlank(positionIdsName, defaults.positionIdsName),
                    nonBlank(logitsName, defaults.logitsName),
                    nonBlank(pastKeyNameTemplate, defaults.pastKeyNameTemplate),
                    nonBlank(pastValueNameTemplate, defaults.pastValueNameTemplate),
                    nonBlank(presentKeyNameTemplate, defaults.presentKeyNameTemplate),
                    nonBlank(presentValueNameTemplate, defaults.presentValueNameTemplate));
        }

        private static String nonBlank(String value, String fallback) {
            return value == null || value.isBlank() ? fallback : value.trim();
        }
    }
}
