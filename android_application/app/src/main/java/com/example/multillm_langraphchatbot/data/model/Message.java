package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;

public class Message {
    @SerializedName("role")    public String role;    // "user" | "assistant"
    @SerializedName("content") public String content;

    public boolean isUser() { return "user".equals(role); }
}