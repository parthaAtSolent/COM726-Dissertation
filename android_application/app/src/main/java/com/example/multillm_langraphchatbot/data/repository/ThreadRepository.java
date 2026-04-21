package com.example.multillm_langraphchatbot.data.repository;

import androidx.lifecycle.MutableLiveData;

import com.example.multillm_langraphchatbot.data.model.NewThreadResponse;
import com.example.multillm_langraphchatbot.data.model.Thread;
import com.example.multillm_langraphchatbot.data.model.ThreadsResponse;
import com.example.multillm_langraphchatbot.network.ApiClient;
import com.example.multillm_langraphchatbot.network.ApiService;
import com.example.multillm_langraphchatbot.util.Resource;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * Single source of truth for Thread data.
 * All API calls are fire-and-forget; results are posted to the supplied LiveData.
 *
 * Threading: Retrofit executes callbacks on the main thread via its default
 * executor. No explicit thread-switching needed here.
 */
public class ThreadRepository {

    private final ApiService api;

    public ThreadRepository() {
        api = ApiClient.getClient().create(ApiService.class);
    }

    // ── List Threads ──────────────────────────────────────────────────────────

    public void fetchThreads(MutableLiveData<Resource<List<Thread>>> liveData) {
        liveData.setValue(Resource.loading());
        api.getThreads(50).enqueue(new Callback<ThreadsResponse>() {
            @Override
            public void onResponse(Call<ThreadsResponse> call,
                                   Response<ThreadsResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    liveData.setValue(Resource.success(response.body().threads));
                } else {
                    liveData.setValue(Resource.error("Failed to load threads: HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<ThreadsResponse> call, Throwable t) {
                liveData.setValue(Resource.error("Network error: " + t.getMessage()));
            }
        });
    }

    // ── Create Thread ─────────────────────────────────────────────────────────

    public void createThread(String title, String model,
                             MutableLiveData<Resource<NewThreadResponse>> liveData) {
        liveData.setValue(Resource.loading());
        api.createThread(title, model).enqueue(new Callback<NewThreadResponse>() {
            @Override
            public void onResponse(Call<NewThreadResponse> call,
                                   Response<NewThreadResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    liveData.setValue(Resource.success(response.body()));
                } else {
                    liveData.setValue(Resource.error("Failed to create thread: HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<NewThreadResponse> call, Throwable t) {
                liveData.setValue(Resource.error("Network error: " + t.getMessage()));
            }
        });
    }

    // ── Rename Thread ─────────────────────────────────────────────────────────

    public void renameThread(String threadId, String newTitle,
                             MutableLiveData<Resource<Boolean>> liveData) {
        api.renameThread(threadId, newTitle).enqueue(new Callback<Void>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                if (response.isSuccessful()) {
                    liveData.setValue(Resource.success(true));
                } else {
                    liveData.setValue(Resource.error("Rename failed: HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<Void> call, Throwable t) {
                liveData.setValue(Resource.error("Network error: " + t.getMessage()));
            }
        });
    }

    // ── Delete Thread ─────────────────────────────────────────────────────────

    public void deleteThread(String threadId,
                             MutableLiveData<Resource<Boolean>> liveData) {
        api.deleteThread(threadId).enqueue(new Callback<Void>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                if (response.isSuccessful()) {
                    liveData.setValue(Resource.success(true));
                } else {
                    liveData.setValue(Resource.error("Delete failed: HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<Void> call, Throwable t) {
                liveData.setValue(Resource.error("Network error: " + t.getMessage()));
            }
        });
    }
}