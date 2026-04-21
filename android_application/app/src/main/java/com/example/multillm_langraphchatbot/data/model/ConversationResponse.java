package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class ConversationResponse {
    @SerializedName("thread_id") public String threadId;
    @SerializedName("messages")  public List<Message> messages;
}