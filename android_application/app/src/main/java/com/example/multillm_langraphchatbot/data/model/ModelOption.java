package com.example.multillm_langraphchatbot.data.model;

public class ModelOption {
    private String key;
    private String displayName;
    private String description;
    private boolean isCustom;

    public ModelOption(String key, String displayName, String description, boolean isCustom) {
        this.key = key;
        this.displayName = displayName;
        this.description = description;
        this.isCustom = isCustom;
    }

    public String getKey() { return key; }
    public String getDisplayName() { return displayName; }
    public String getDescription() { return description; }
    public boolean isCustom() { return isCustom; }
}