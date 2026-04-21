package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;

public class UploadResponse {
    @SerializedName("status")   public String status;
    @SerializedName("filename") public String filename;
    @SerializedName("chunks")   public int chunks;
}