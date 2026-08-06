package tech.kayys.gollek.sdk.internal.util;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

/**
 * Internal JSON utility wrapping Jackson's {@link ObjectMapper}.
 *
 * <p>Configured with sensible defaults:
 * <ul>
 *   <li>Unknown properties are silently ignored during deserialization.</li>
 *   <li>Dates are serialized as ISO-8601 strings, not timestamps.</li>
 *   <li>{@code null} fields are omitted from serialized output.</li>
 * </ul>
 *
 * <p>This class is package-private in intent; it is {@code public} only to allow
 * access from sibling internal packages.
 */
public final class Json {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false)
            .setSerializationInclusion(JsonInclude.Include.NON_NULL);

    private Json() {
    }

    /**
     * Serializes an object to a JSON string.
     *
     * @param obj the object to serialize; must not be {@code null}
     * @return the JSON representation
     * @throws RuntimeException if serialization fails
     */
    public static String stringify(Object obj) {
        try {
            return MAPPER.writeValueAsString(obj);
        } catch (Exception e) {
            throw new RuntimeException("Failed to encode JSON", e);
        }
    }

    /**
     * Deserializes a JSON string into an instance of the given class.
     *
     * @param <T>   the target type
     * @param json  the JSON string to parse
     * @param clazz the target class
     * @return the deserialized object
     * @throws RuntimeException if parsing fails
     */
    public static <T> T parse(String json, Class<T> clazz) {
        try {
            return MAPPER.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException("Failed to decode JSON: " + json, e);
        }
    }

    /**
     * Deserializes a JSON array string into a {@link java.util.List} of the given type.
     *
     * @param <T>   the element type
     * @param json  the JSON array string
     * @param clazz the element class
     * @return a list of deserialized objects
     * @throws RuntimeException if parsing fails
     */
    public static <T> java.util.List<T> parseList(String json, Class<T> clazz) {
        try {
            return MAPPER.readValue(json, MAPPER.getTypeFactory().constructCollectionType(java.util.List.class, clazz));
        } catch (Exception e) {
            throw new RuntimeException("Failed to decode JSON list: " + json, e);
        }
    }

    /**
     * Alias for {@link #stringify(Object)}.
     *
     * @param obj the object to encode
     * @return the JSON string
     */
    public static String encode(Object obj) {
        return stringify(obj);
    }

    /**
     * Alias for {@link #parse(String, Class)}.
     *
     * @param <T>   the target type
     * @param json  the JSON string
     * @param clazz the target class
     * @return the deserialized object
     */
    public static <T> T decode(String json, Class<T> clazz) {
        return parse(json, clazz);
    }
}
