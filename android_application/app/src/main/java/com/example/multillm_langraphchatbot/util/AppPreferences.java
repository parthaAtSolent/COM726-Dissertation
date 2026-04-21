package com.example.multillm_langraphchatbot.util;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Thin SharedPreferences wrapper for persisting app-level state:
 * - last active thread ID
 * - selected model
 * - server base URL (for runtime override)
 */
public class AppPreferences {

    private static final String PREFS_NAME    = "langgraph_prefs";
    private static final String KEY_THREAD_ID = "last_thread_id";
    private static final String KEY_MODEL     = "selected_model";
    private static final String KEY_SERVER    = "server_base_url";

    public static final String DEFAULT_MODEL = "llama-8b-instant";

    private final SharedPreferences prefs;

    public AppPreferences(Context context) {
        prefs = context.getApplicationContext()
                .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    public String getLastThreadId()            { return prefs.getString(KEY_THREAD_ID, null); }
    public void   setLastThreadId(String id)   { prefs.edit().putString(KEY_THREAD_ID, id).apply(); }

    public String getSelectedModel()           { return prefs.getString(KEY_MODEL, DEFAULT_MODEL); }
    public void   setSelectedModel(String key) { prefs.edit().putString(KEY_MODEL, key).apply(); }

    public String getServerUrl()               { return prefs.getString(KEY_SERVER, null); }
    public void   setServerUrl(String url)     { prefs.edit().putString(KEY_SERVER, url).apply(); }
}