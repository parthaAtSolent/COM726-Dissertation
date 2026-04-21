package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;

public class ChatRequest {
    @SerializedName("thread_id") public final String threadId;
    @SerializedName("message")   public final String message;
    @SerializedName("model")     public final String model;

    public ChatRequest(String threadId, String message, String model) {
        this.threadId = threadId;
        this.message  = message;
        this.model    = model;
    }
}