package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;

public class NewThreadResponse {
    @SerializedName("thread_id") public String threadId;
    @SerializedName("title")     public String title;
}