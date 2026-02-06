package com.kmpchatbot.server.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "conversations")
@Data
@EqualsAndHashCode(callSuper = false)
public class Conversation extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String title;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @OneToMany(mappedBy = "conversation", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("timestamp ASC")
    private List<Message> messages = new ArrayList<>();

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Column(name = "is_archived")
    private boolean archived = false;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (title == null || title.isEmpty()) {
            title = "Conversation " + LocalDateTime.now().toString();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Custom finder methods
    public static List<Conversation> findByUser(Long userId) {
        return list("user.id = ?1 and archived = false", userId);
    }

    public static List<Conversation> findByUserIncludingArchived(Long userId) {
        return list("user.id", userId);
    }

    public static Conversation findByIdAndUser(Long id, Long userId) {
        return find("id = ?1 and user.id = ?2", id, userId).firstResult();
    }
}
