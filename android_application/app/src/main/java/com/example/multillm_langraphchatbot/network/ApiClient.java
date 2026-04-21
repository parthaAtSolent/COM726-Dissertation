package com.example.multillm_langraphchatbot.network;

import com.example.multillm_langraphchatbot.BuildConfig;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * Retrofit singleton with configurable base URL from BuildConfig.
 *
 * RISK NOTED: BASE_URL is baked at compile time from BuildConfig.
 * For a production app with a Settings screen, swap to a runtime-configurable
 * approach using resetClient() below when the user changes the server URL.
 */
public class ApiClient {

    private static volatile Retrofit retrofit;
    private static String currentBaseUrl = BuildConfig.SERVER_BASE_URL;

    private ApiClient() {}

    public static Retrofit getClient() {
        if (retrofit == null) {
            synchronized (ApiClient.class) {
                if (retrofit == null) {
                    retrofit = buildRetrofit(currentBaseUrl);
                }
            }
        }
        return retrofit;
    }

    /**
     * Call when the user changes the server URL in Settings.
     * Forces a new Retrofit instance on next getClient() call.
     */
    public static synchronized void resetClient(String newBaseUrl) {
        currentBaseUrl = newBaseUrl.endsWith("/") ? newBaseUrl : newBaseUrl + "/";
        retrofit = null;
    }

    private static Retrofit buildRetrofit(String baseUrl) {
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(BuildConfig.DEBUG
                ? HttpLoggingInterceptor.Level.BODY
                : HttpLoggingInterceptor.Level.NONE);

        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)   // LLM replies can be slow
                .writeTimeout(30, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build();

        return new Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }
}