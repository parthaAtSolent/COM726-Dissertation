package com.example.multillm_langraphchatbot.data.model;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class FilesResponse {
    @SerializedName("files") public List<String> files;
}