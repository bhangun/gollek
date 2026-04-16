/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.tooling;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.*;

import java.util.Optional;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for {@link ConversationSessionManager} — session lifecycle,
 * multi-turn token accumulation, and expiry behaviour.
 */
@QuarkusTest
class ConversationSessionManagerTest {

    @Inject
    ConversationSessionManager sessionMgr;

    private static final String MODEL_KEY = "/models/test-model";
    private static final String SESSION_ID = "test-session-001";
    private static final int[] INITIAL_IDS = { 1, 42, 100, 200, 300 }; // fake token IDs

    @BeforeEach
    void cleanup() {
        sessionMgr.close(SESSION_ID);
    }

    // ── Creation ──────────────────────────────────────────────────────────────

    @Test
    @DisplayName("create: session is findable immediately after creation")
    void testCreateAndFind() {
        sessionMgr.create(SESSION_ID, MODEL_KEY, INITIAL_IDS, null);

        Optional<ConversationSessionManager.ConversationSession> found = sessionMgr.find(SESSION_ID, MODEL_KEY);

        assertThat(found).isPresent();
        assertThat(found.get().sessionId()).isEqualTo(SESSION_ID);
        assertThat(found.get().modelKey()).isEqualTo(MODEL_KEY);
        assertThat(found.get().totalTokens()).isEqualTo(INITIAL_IDS.length);
    }

    // ── Find by wrong model key ───────────────────────────────────────────────

    @Test
    @DisplayName("find: wrong model key returns empty")
    void testFindWrongModelKey() {
        sessionMgr.create(SESSION_ID, MODEL_KEY, INITIAL_IDS, null);

        Optional<ConversationSessionManager.ConversationSession> found = sessionMgr.find(SESSION_ID,
                "/models/different-model");

        assertThat(found).isEmpty();
    }

    // ── Missing session ───────────────────────────────────────────────────────

    @Test
    @DisplayName("find: unknown session ID returns empty")
    void testFindUnknown() {
        assertThat(sessionMgr.find("nonexistent-xyz", MODEL_KEY)).isEmpty();
    }

    @Test
    @DisplayName("find: null session ID returns empty")
    void testFindNull() {
        assertThat(sessionMgr.find(null, MODEL_KEY)).isEmpty();
        assertThat(sessionMgr.find("  ", MODEL_KEY)).isEmpty();
    }

    // ── Token extension ───────────────────────────────────────────────────────

    @Test
    @DisplayName("extend: appends new tokens to session")
    void testExtend() {
        sessionMgr.create(SESSION_ID, MODEL_KEY, INITIAL_IDS, null);

        int[] newTokens = { 400, 500, 600 };
        sessionMgr.extend(SESSION_ID, newTokens);

        Optional<ConversationSessionManager.ConversationSession> found = sessionMgr.find(SESSION_ID, MODEL_KEY);

        assertThat(found).isPresent();
        assertThat(found.get().totalTokens())
                .isEqualTo(INITIAL_IDS.length + newTokens.length);

        // extendWith should include both existing and new tokens
        int[] combined = found.get().extendWith(new int[] { 700, 800 });
        assertThat(combined).hasSize(INITIAL_IDS.length + newTokens.length + 2);
    }

    // ── Close ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("close: removes session from registry")
    void testClose() {
        sessionMgr.create(SESSION_ID, MODEL_KEY, INITIAL_IDS, null);
        assertThat(sessionMgr.find(SESSION_ID, MODEL_KEY)).isPresent();

        sessionMgr.close(SESSION_ID);

        assertThat(sessionMgr.find(SESSION_ID, MODEL_KEY)).isEmpty();
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("activeSessions: increases on create, decreases on close")
    void testStats() {
        int before = sessionMgr.activeSessions();

        sessionMgr.create(SESSION_ID + "-a", MODEL_KEY, INITIAL_IDS, null);
        sessionMgr.create(SESSION_ID + "-b", MODEL_KEY, INITIAL_IDS, null);
        assertThat(sessionMgr.activeSessions()).isEqualTo(before + 2);

        sessionMgr.close(SESSION_ID + "-a");
        assertThat(sessionMgr.activeSessions()).isEqualTo(before + 1);

        sessionMgr.close(SESSION_ID + "-b");
        assertThat(sessionMgr.activeSessions()).isEqualTo(before);
    }

    @Test
    @DisplayName("totalCreated: monotonically increases")
    void testTotalCreated() {
        int before = sessionMgr.totalCreated();
        sessionMgr.create(SESSION_ID + "-c", MODEL_KEY, INITIAL_IDS, null);
        assertThat(sessionMgr.totalCreated()).isEqualTo(before + 1);
        sessionMgr.close(SESSION_ID + "-c");
    }

    // ── extendWith helper ─────────────────────────────────────────────────────

    @Test
    @DisplayName("ConversationSession.extendWith: produces correct combined array")
    void testExtendWith() {
        sessionMgr.create(SESSION_ID, MODEL_KEY, new int[] { 1, 2, 3 }, null);
        var session = sessionMgr.find(SESSION_ID, MODEL_KEY).orElseThrow();

        int[] delta = { 4, 5, 6 };
        int[] combined = session.extendWith(delta);

        assertThat(combined).containsExactly(1, 2, 3, 4, 5, 6);
        // Original session token IDs should be unchanged
        assertThat(session.totalTokens()).isEqualTo(3);
    }

    // ── truncateToHalf (context window management) ────────────────────────────

    @Test
    @DisplayName("truncateToHalf: halves the accumulated token count")
    void testTruncateToHalf() {
        sessionMgr.create(SESSION_ID, MODEL_KEY, new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }, null);
        var session = sessionMgr.find(SESSION_ID, MODEL_KEY).orElseThrow();

        session.truncateToHalf();

        // 8 tokens → 4 tokens remaining (most recent half)
        assertThat(session.totalTokens()).isEqualTo(4);
    }
}
