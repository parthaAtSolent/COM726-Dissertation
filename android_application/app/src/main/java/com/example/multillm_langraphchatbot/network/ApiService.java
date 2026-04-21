package com.example.multillm_langraphchatbot.network;

import com.example.multillm_langraphchatbot.data.model.*;

import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.*;

public interface ApiService {

    // ── Health ──────────────────────────────────────────────────────────────
    @GET("health")
    Call<HealthResponse> checkHealth();

    // ── Chat ─────────────────────────────────────────────────────────────────
    @POST("chat")
    Call<ChatResponse> sendMessage(@Body ChatRequest request);

    // ── Threads ───────────────────────────────────────────────────────────────
    @GET("threads")
    Call<ThreadsResponse> getThreads(@Query("limit") int limit);

    @POST("threads")
    Call<NewThreadResponse> createThread(
            @Query("title") String title,
            @Query("model") String model);

    @GET("threads/{thread_id}")
    Call<ThreadDetailResponse> getThreadDetail(@Path("thread_id") String threadId);

    @GET("threads/{thread_id}/messages")
    Call<ConversationResponse> getMessages(@Path("thread_id") String threadId);

    @PUT("threads/{thread_id}/title")
    Call<Void> renameThread(
            @Path("thread_id") String threadId,
            @Query("title") String title);

    @DELETE("threads/{thread_id}")
    Call<Void> deleteThread(@Path("thread_id") String threadId);

    // ── RAG ───────────────────────────────────────────────────────────────────
    @Multipart
    @POST("upload")
    Call<UploadResponse> uploadFile(@Part MultipartBody.Part file);

    @GET("files")
    Call<FilesResponse> getFiles();

    // ── Inner Response Classes ─────────────────────────────────────────────
    class HealthResponse {
        public String status;
        public String service;
    }

    class ThreadDetailResponse {
        public String thread_id;
        public String title;
        public String model;
        public String created_at;
        public java.util.List<com.example.multillm_langraphchatbot.data.model.Message> messages;
    }
}