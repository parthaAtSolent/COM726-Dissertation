package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;

public class Thread {
    @SerializedName("thread_id") public String threadId;
    @SerializedName("title")     public String title;
    @SerializedName("model")     public String model;
    @SerializedName("created_at") public String createdAt;
}