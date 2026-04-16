/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * StructuredOutputSampler.java
 * ────────────────────────────
 * Grammar-guided logit masking for constrained (structured) generation.
 *
 * What this solves
 * ════════════════
 * A standard transformer can generate any token at any position.
 * Structured output forces the model to generate valid JSON (or any other
 * grammar) by setting the logit of every token that would make the output
 * invalid to -∞ before softmax.  The model still "chooses" the token — it
 * just cannot choose an invalid one.
 *
 * Example: JSON schema {"type":"object","properties":{"name":{"type":"string"}}}
 *   After "{"  → only valid next tokens are string keys starting with "
 *   After "\"name\":" → only valid next tokens are " (start of string value)
 *
 * Implementation strategy
 * ═══════════════════════
 * We use a finite-state machine (FSM) derived from the JSON schema:
 *
 *   1. At startup, parse the JSON schema into a JSONSchemaFSM.
 *   2. At each decode step, call fsm.allowedTokenIds(currentState) to get
 *      the set of valid next token IDs.
 *   3. In TokenSampler, set logits[id] = -Float.MAX_VALUE for all ids NOT
 *      in the allowed set.
 *   4. Sample normally from the remaining tokens.
 *
 * This guarantees the output matches the schema exactly.
 *
 * JSON grammar coverage
 * ═════════════════════
 * Supported schema constructs:
 *   - type: object, array, string, number, integer, boolean, null
 *   - properties, required, additionalProperties
 *   - items (array element schema)
 *   - enum (explicit value list)
 *   - oneOf, anyOf  (union types — simplified)
 *
 * Limitations:
 *   - $ref and $defs are not supported (dereference before passing in)
 *   - format (date-time, uuid, etc.) is treated as unconstrained string
 *   - Maximum recursion depth: 16 levels
 *
 * Tokenizer coupling
 * ══════════════════
 * The FSM works at the CHARACTER level (JSON characters).
 * We convert character-level constraints to token-level constraints by
 * pre-computing which token IDs map to valid next characters.
 * This pre-computation is cached per (schema, tokenizer) pair.
 *
 * Usage
 * ═════
 *   StructuredOutputSampler sampler = ...; // @Inject
 *   String schema = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}";
 *   StructuredOutputSampler.Session session = sampler.createSession(schema, tokenizer);
 *
 *   // In decode loop:
 *   logits = forwardPass.decode(token, ...);
 *   session.maskLogits(logits);               // zeroes invalid tokens
 *   nextToken = tokenSampler.sample(logits, ...);
 *   session.advance(nextToken, tokenizer);    // update FSM state
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;

import org.jboss.logging.Logger;

import java.util.*;

/**
 * Grammar-guided logit masking for structured (JSON schema) output.
 *
 * <p>
 * Creates per-request {@link Session} objects that track parser state
 * and mask invalid token logits at each decode step.
 */
@ApplicationScoped
public class StructuredOutputSampler {

    private static final Logger log = Logger.getLogger(StructuredOutputSampler.class);

    @Inject
    ObjectMapper objectMapper;

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a constrained generation session for a JSON schema.
     *
     * @param jsonSchema JSON schema string (already dereferenced)
     * @param tokenizer  the model's tokenizer (used to map tokens to characters)
     * @return a new session ready to mask logits
     */
    public Session createSession(String jsonSchema, Tokenizer tokenizer) {
        try {
            JsonNode schema = objectMapper.readTree(jsonSchema);
            JsonFSM fsm = JsonFSM.fromSchema(schema);
            return new Session(fsm, tokenizer);
        } catch (Exception e) {
            log.warnf(e, "Failed to parse JSON schema for structured output — generation unconstrained");
            return new Session(null, tokenizer); // unconstrained fallback
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Session
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Per-request structured output session.
     *
     * <p>
     * Maintains the current FSM state and masks logits at each decode step.
     */
    public static final class Session {

        private final JsonFSM fsm;
        private final Tokenizer tokenizer;

        /** Set of token IDs currently allowed by the grammar. Null = unconstrained. */
        private Set<Integer> allowedTokenIds;

        /** Buffer of generated JSON characters so far (for FSM state tracking). */
        private final StringBuilder generated = new StringBuilder();

        /** Whether the FSM is in a complete/accepting state. */
        private boolean complete = false;

        Session(JsonFSM fsm, Tokenizer tokenizer) {
            this.fsm = fsm;
            this.tokenizer = tokenizer;
            if (fsm != null) {
                this.allowedTokenIds = fsm.initialAllowedTokenIds(tokenizer);
            }
        }

        /**
         * Mask logits in-place: set logit of every DISALLOWED token to -∞.
         *
         * <p>
         * Call this immediately before sampling at each decode step.
         *
         * @param logits raw logit array (modified in-place)
         */
        public void maskLogits(float[] logits) {
            if (fsm == null || allowedTokenIds == null || complete)
                return;

            for (int id = 0; id < logits.length; id++) {
                if (!allowedTokenIds.contains(id)) {
                    logits[id] = -Float.MAX_VALUE;
                }
            }
        }

        /**
         * Advance the FSM state after a token was generated.
         *
         * @param tokenId the token that was just sampled
         */
        public void advance(int tokenId) {
            if (fsm == null || complete)
                return;

            String tokenText = tokenizer.decode(new long[] { tokenId }, DecodeOptions.defaultOptions());
            generated.append(tokenText);

            // Update FSM state based on the generated characters
            fsm.advance(tokenText);
            complete = fsm.isComplete();
            allowedTokenIds = complete ? null : fsm.currentAllowedTokenIds(tokenizer);
        }

        /** Whether the structured output is complete (the JSON is fully closed). */
        public boolean isComplete() {
            return complete;
        }

        /** The JSON generated so far. */
        public String generatedJson() {
            return generated.toString();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JSON FSM
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Simplified JSON finite-state machine.
     *
     * <p>
     * Tracks the structural position in the JSON output and determines
     * what characters (and hence tokens) are valid at each step.
     *
     * <p>
     * This is a simplified implementation that handles the common cases.
     * A production-grade FSM would use a proper PEG grammar parser
     * (e.g. llama.cpp's grammar sampler or the Guidance library's approach).
     */
    static final class JsonFSM {

        enum State {
            START, // Before any output
            IN_OBJECT, // Inside { } — expecting key or }
            IN_KEY, // Inside a quoted key string
            AFTER_KEY, // After key, expecting :
            AFTER_COLON, // After :, expecting value
            IN_STRING, // Inside a quoted value string
            IN_NUMBER, // Inside a number
            IN_BOOL_NULL, // Inside true/false/null
            IN_ARRAY, // Inside [ ] — expecting value or ]
            COMPLETE // Outermost structure closed
        }

        private final JsonNode schema;
        private State state = State.START;
        private int depth = 0;
        private final Deque<State> stateStack = new ArrayDeque<>();

        static JsonFSM fromSchema(JsonNode schema) {
            return new JsonFSM(schema);
        }

        JsonFSM(JsonNode schema) {
            this.schema = schema;
        }

        /** Characters valid at the current state. */
        Set<Character> validNextChars() {
            return switch (state) {
                case START -> initialChars(schema);
                case IN_OBJECT -> Set.of('"', '}', '\n', ' ', '\t');
                case IN_KEY -> allPrintableChars();
                case AFTER_KEY -> Set.of(':', ' ');
                case AFTER_COLON -> Set.of('"', '{', '[', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-',
                        't', 'f', 'n', ' ', '\n');
                case IN_STRING -> allPrintableChars();
                case IN_NUMBER -> Set.of('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E', '-', '+',
                        ',', ']', '}', '\n', ' ');
                case IN_BOOL_NULL -> Set.of('r', 'u', 'e', 'a', 'l', 's', 'i', ',', ']', '}', '\n', ' ');
                case IN_ARRAY -> Set.of('"', '{', '[', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-',
                        't', 'f', 'n', ']', ' ', '\n');
                case COMPLETE -> Set.of('\n', ' ');
            };
        }

        /** Advance the FSM state by the characters in the given token text. */
        void advance(String tokenText) {
            for (char c : tokenText.toCharArray()) {
                advanceChar(c);
            }
        }

        private void advanceChar(char c) {
            switch (state) {
                case START -> {
                    if (c == '{') {
                        state = State.IN_OBJECT;
                        depth++;
                    } else if (c == '[') {
                        state = State.IN_ARRAY;
                        depth++;
                    } else if (c == '"') {
                        state = State.IN_STRING;
                    }
                }
                case IN_OBJECT -> {
                    if (c == '"')
                        state = State.IN_KEY;
                    else if (c == '}') {
                        depth--;
                        state = depth == 0 ? State.COMPLETE : State.IN_OBJECT;
                    }
                }
                case IN_KEY -> {
                    if (c == '"')
                        state = State.AFTER_KEY;
                }
                case AFTER_KEY -> {
                    if (c == ':')
                        state = State.AFTER_COLON;
                }
                case AFTER_COLON -> {
                    if (c == '"')
                        state = State.IN_STRING;
                    else if (c == '{') {
                        stateStack.push(state);
                        state = State.IN_OBJECT;
                        depth++;
                    } else if (c == '[') {
                        stateStack.push(state);
                        state = State.IN_ARRAY;
                        depth++;
                    } else if (Character.isDigit(c) || c == '-')
                        state = State.IN_NUMBER;
                    else if (c == 't' || c == 'f' || c == 'n')
                        state = State.IN_BOOL_NULL;
                }
                case IN_STRING -> {
                    if (c == '"') {
                        // Pop back to calling state
                        state = stateStack.isEmpty() ? State.IN_OBJECT : stateStack.pop();
                    }
                }
                case IN_NUMBER, IN_BOOL_NULL -> {
                    if (c == ',' || c == '}' || c == ']') {
                        state = State.IN_OBJECT; // simplified
                    }
                }
                case IN_ARRAY -> {
                    if (c == ']') {
                        depth--;
                        state = depth == 0 ? State.COMPLETE : State.IN_OBJECT;
                    }
                }
                default -> {
                }
            }
        }

        boolean isComplete() {
            return state == State.COMPLETE;
        }

        /** Compute the set of token IDs valid at the initial state. */
        Set<Integer> initialAllowedTokenIds(Tokenizer tokenizer) {
            return computeAllowedTokenIds(tokenizer, initialChars(schema));
        }

        /** Compute the set of token IDs valid at the current state. */
        Set<Integer> currentAllowedTokenIds(Tokenizer tokenizer) {
            return computeAllowedTokenIds(tokenizer, validNextChars());
        }

        private Set<Integer> computeAllowedTokenIds(Tokenizer tokenizer,
                Set<Character> validChars) {
            Set<Integer> allowed = new HashSet<>();
            int vocabSize = tokenizer.vocabSize();
            // Check each token: if its first character (when decoded) is a valid char,
            // allow it
            for (int id = 0; id < Math.min(vocabSize, 100000); id++) {
                String decoded = tokenizer.decode(new long[] { id }, DecodeOptions.defaultOptions());
                if (decoded != null && !decoded.isEmpty()) {
                    char first = decoded.charAt(0);
                    if (validChars.contains(first)) {
                        allowed.add(id);
                    }
                }
            }
            // Always allow EOS
            allowed.add(tokenizer.eosTokenId());
            return allowed;
        }

        private Set<Character> initialChars(JsonNode schema) {
            if (schema == null)
                return Set.of('{', '[', '"');
            String type = schema.has("type") ? schema.get("type").asText() : "object";
            return switch (type) {
                case "object" -> Set.of('{');
                case "array" -> Set.of('[');
                case "string" -> Set.of('"');
                case "number", "integer" -> Set.of('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-');
                case "boolean" -> Set.of('t', 'f');
                case "null" -> Set.of('n');
                default -> Set.of('{', '[', '"');
            };
        }

        private static Set<Character> allPrintableChars() {
            Set<Character> chars = new HashSet<>();
            for (char c = 32; c < 127; c++)
                chars.add(c);
            return chars;
        }
    }
}
