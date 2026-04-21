package com.example.multillm_langraphchatbot;

import android.app.Application;
import com.example.multillm_langraphchatbot.data.repository.ChatRepository;
import com.example.multillm_langraphchatbot.data.repository.ThreadRepository;

/**
 * Application class — initialises singletons at startup.
 */
public class LangGraphApp extends Application {

    private static LangGraphApp instance;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
    }

    public static LangGraphApp getInstance() {
        return instance;
    }
}