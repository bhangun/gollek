package com.kmpchatbot.server.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Entity
@Table(name = "messages")
@Data
@EqualsAndHashCode(callSuper = false)
public class Message extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "conversation_id", nullable = false)
    private Conversation conversation;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private MessageRole role;

    @Column(nullable = false, updatable = false)
    private LocalDateTime timestamp;

    @Column(name = "token_count")
    private Integer tokenCount;

    @Column(name = "model_used")
    private String modelUsed;

    @PrePersist
    protected void onCreate() {
        if (timestamp == null) {
            timestamp = LocalDateTime.now();
        }
    }

    public enum MessageRole {
        USER,
        ASSISTANT,
        SYSTEM
    }

    // Custom finder methods
    public static List<Message> findByConversation(Long conversationId) {
        return list("conversation.id = ?1 order by timestamp asc", conversationId);
    }

    public static List<Message> findRecentByConversation(Long conversationId, int limit) {
        return find("conversation.id = ?1 order by timestamp desc", conversationId)
                .page(0, limit)
                .list();
    }
}
