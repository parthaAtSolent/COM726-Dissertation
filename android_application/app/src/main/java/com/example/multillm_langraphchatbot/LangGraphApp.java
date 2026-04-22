package com.example.multillm_langraphchatbot;

import android.app.Application;
import android.util.Log;

public class LangGraphApp extends Application {
    private static final String TAG = "LangGraphApp";
    private static LangGraphApp instance;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;

        // Catch uncaught exceptions
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            Log.e(TAG, "Uncaught exception: ", throwable);
            // You can also show a dialog here
        });
    }

    public static LangGraphApp getInstance() {
        return instance;
    }
}