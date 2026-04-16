package com.example.multillm_langraphchatbot.network;

import retrofit2.Call;
import retrofit2.http.*;
import java.util.List;

public interface ApiService {

    @GET("health")
    Call<HealthResponse> checkHealth();

    @POST("threads")
    Call<ThreadResponse> createThread(
            @Query("title") String title,
            @Query("model") String model
    );

    @POST("chat")
    Call<ChatResponse> sendMessage(@Body ChatRequest request);

    @GET("threads/{thread_id}/messages")
    Call<ConversationResponse> getConversationHistory(@Path("thread_id") String threadId);

    // Response Models
    class HealthResponse {
        public String status;
        public String service;
    }

    class ThreadResponse {
        public String thread_id;
        public String title;
    }

    class ChatRequest {
        public String thread_id;
        public String message;
        public String model;

        public ChatRequest(String thread_id, String message, String model) {
            this.thread_id = thread_id;
            this.message = message;
            this.model = model;
        }
    }

    class ChatResponse {
        public String response;
        public String thread_id;
    }

    class ConversationResponse {
        public String thread_id;
        public List<MessageItem> messages;
    }

    class MessageItem {
        public String role;
        public String content;
    }
}