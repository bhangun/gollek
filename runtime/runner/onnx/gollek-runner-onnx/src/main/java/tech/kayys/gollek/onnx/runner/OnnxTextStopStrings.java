package tech.kayys.gollek.onnx.runner;

import java.lang.reflect.Array;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import tech.kayys.gollek.spi.inference.InferenceRequest;

final class OnnxTextStopStrings {

    private static final OnnxTextStopStrings EMPTY = new OnnxTextStopStrings(List.of());

    private static final String[] PARAMETER_KEYS = {
            "stop",
            "stop_strings",
            "stopStrings",
            "stop_sequences",
            "stopSequences"
    };

    private final List<String> values;
    private final int minLength;
    private final int maxLength;

    private OnnxTextStopStrings(List<String> values) {
        this.values = values;
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (String value : values) {
            int length = value.length();
            min = Math.min(min, length);
            max = Math.max(max, length);
        }
        this.minLength = values.isEmpty() ? 0 : min;
        this.maxLength = max;
    }

    static OnnxTextStopStrings from(InferenceRequest request) {
        return fromParameters(request == null ? null : request.getParameters());
    }

    static OnnxTextStopStrings fromParameters(Map<String, Object> parameters) {
        if (parameters == null || parameters.isEmpty()) {
            return EMPTY;
        }
        Set<String> stops = new LinkedHashSet<>();
        for (String key : PARAMETER_KEYS) {
            addValue(stops, parameters.get(key));
        }
        return stops.isEmpty() ? EMPTY : new OnnxTextStopStrings(List.copyOf(stops));
    }

    boolean isEmpty() {
        return values.isEmpty();
    }

    boolean matches(String text) {
        if (text == null || text.length() < minLength) {
            return false;
        }
        for (String stop : values) {
            if (text.length() >= stop.length() && text.endsWith(stop)) {
                return true;
            }
        }
        return false;
    }

    int maxLength() {
        return maxLength;
    }

    List<String> values() {
        return values;
    }

    private static void addValue(Set<String> stops, Object value) {
        if (value == null) {
            return;
        }
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                addSingle(stops, item);
            }
            return;
        }
        Class<?> type = value.getClass();
        if (type.isArray()) {
            int length = Array.getLength(value);
            for (int i = 0; i < length; i++) {
                addSingle(stops, Array.get(value, i));
            }
            return;
        }
        addSingle(stops, value);
    }

    private static void addSingle(Set<String> stops, Object value) {
        if (value == null) {
            return;
        }
        String text = value.toString();
        if (!text.isEmpty()) {
            stops.add(text);
        }
    }
}
