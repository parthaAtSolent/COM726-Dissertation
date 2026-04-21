package com.example.multillm_langraphchatbot.util;

import com.example.multillm_langraphchatbot.data.model.ModelOption;

import java.util.ArrayList;
import java.util.List;

public class ModelHelper {

    public static List<ModelOption> getAvailableModels() {
        List<ModelOption> models = new ArrayList<>();

        // Custom router (auto-selects best model)
        models.add(new ModelOption("custom", "🤖 Custom (Auto-router)",
                "Automatically routes to the best model for your query", true));

        // Individual models
        models.add(new ModelOption("deepseek_r1", "🧠 DeepSeek R1",
                "Advanced reasoning model", false));
        models.add(new ModelOption("falcon3", "🦅 Falcon3",
                "Fast and efficient", false));
        models.add(new ModelOption("gemini_2_5_flash", "✨ Gemini 2.5 Flash",
                "Google's fast model", false));
        models.add(new ModelOption("gemma3_270m", "🔬 Gemma3 270M",
                "Lightweight Google model", false));
        models.add(new ModelOption("granite3_dense_2b", "🗿 Granite3 Dense 2B",
                "IBM's efficient model", false));
        models.add(new ModelOption("llama_3_1_8b_instant", "🦙 Llama 3.1 8B Instant",
                "Meta's fast Llama model", false));
        models.add(new ModelOption("mistral_7b", "🌬️ Mistral 7B",
                "High performance 7B model", false));
        models.add(new ModelOption("phi3_3_8b", "📘 Phi3 3.8B",
                "Microsoft's compact model", false));
        models.add(new ModelOption("qwen2_5_coder_7b", "🐉 Qwen2.5 Coder 7B",
                "Specialized for coding", false));
        models.add(new ModelOption("qwen3_5_0_8b", "🐉 Qwen3.5 0.8B",
                "Ultra-lightweight model", false));

        return models;
    }

    public static String getModelKeyForApi(String selectedKey) {
        // If custom is selected, send "custom" to API (your backend will route it)
        // Otherwise send the actual model key
        return selectedKey;
    }
}