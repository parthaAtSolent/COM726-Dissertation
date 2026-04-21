package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;

public class ChatResponse {
    @SerializedName("response")  public String response;
    @SerializedName("thread_id") public String threadId;
}