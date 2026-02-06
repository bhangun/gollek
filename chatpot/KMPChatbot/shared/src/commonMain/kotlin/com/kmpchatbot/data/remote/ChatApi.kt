package com.kmpchatbot.data.remote

import com.kmpchatbot.data.remote.dto.ChatRequest
import com.kmpchatbot.data.remote.dto.ChatResponse
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.logging.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.Json

interface ChatApi {
    suspend fun sendMessage(request: ChatRequest): Result<ChatResponse>
}

class ChatApiImpl(
    private val apiKey: String = "your-api-key-here"
) : ChatApi {
    
    private val client = HttpClient {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
            })
        }
        install(Logging) {
            level = LogLevel.INFO
        }
    }
    
    override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
        return try {
            val response = client.post("https://api.anthropic.com/v1/messages") {
                contentType(ContentType.Application.Json)
                header("x-api-key", apiKey)
                header("anthropic-version", "2023-06-01")
                setBody(request)
            }
            Result.success(response.body())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    fun close() {
        client.close()
    }
}
