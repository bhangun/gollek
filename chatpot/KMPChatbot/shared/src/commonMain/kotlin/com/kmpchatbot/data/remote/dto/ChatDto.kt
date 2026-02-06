package com.kmpchatbot.data.remote.dto

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ChatRequest(
    val model: String = "claude-sonnet-4-20250514",
    val messages: List<MessageDto>,
    @SerialName("max_tokens")
    val maxTokens: Int = 1024
)

@Serializable
data class MessageDto(
    val role: String,
    val content: String
)

@Serializable
data class ChatResponse(
    val id: String,
    val type: String,
    val role: String,
    val content: List<ContentBlock>,
    val model: String,
    @SerialName("stop_reason")
    val stopReason: String? = null
)

@Serializable
data class ContentBlock(
    val type: String,
    val text: String? = null
)
