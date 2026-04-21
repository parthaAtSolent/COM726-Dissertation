package com.example.multillm_langraphchatbot.data.repository;

import androidx.lifecycle.MutableLiveData;

import com.example.multillm_langraphchatbot.data.model.ChatRequest;
import com.example.multillm_langraphchatbot.data.model.ChatResponse;
import com.example.multillm_langraphchatbot.data.model.ConversationResponse;
import com.example.multillm_langraphchatbot.data.model.Message;
import com.example.multillm_langraphchatbot.network.ApiClient;
import com.example.multillm_langraphchatbot.network.ApiService;
import com.example.multillm_langraphchatbot.util.Resource;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ChatRepository {

    private final ApiService api;

    public ChatRepository() {
        api = ApiClient.getClient().create(ApiService.class);
    }

    public void loadMessages(String threadId,
                             MutableLiveData<Resource<List<Message>>> liveData) {
        liveData.setValue(Resource.loading());
        api.getMessages(threadId).enqueue(new Callback<ConversationResponse>() {
            @Override
            public void onResponse(Call<ConversationResponse> call,
                                   Response<ConversationResponse> response) {
                android.util.Log.d("ChatRepository", "loadMessages response code: " + response.code());
                if (response.isSuccessful() && response.body() != null) {
                    liveData.setValue(Resource.success(response.body().messages));
                } else {
                    liveData.setValue(Resource.error("Failed to load messages: HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<ConversationResponse> call, Throwable t) {
                android.util.Log.e("ChatRepository", "loadMessages error: " + t.getMessage());
                liveData.setValue(Resource.error("Network error: " + t.getMessage()));
            }
        });
    }

    public void sendMessage(String threadId, String message, String model,
                            MutableLiveData<Resource<ChatResponse>> liveData) {
        liveData.setValue(Resource.loading());
        ChatRequest request = new ChatRequest(threadId, message, model);

        android.util.Log.d("ChatRepository", "Sending message - ThreadId: " + threadId + ", Model: " + model + ", Message: " + message);

        api.sendMessage(request).enqueue(new Callback<ChatResponse>() {
            @Override
            public void onResponse(Call<ChatResponse> call,
                                   Response<ChatResponse> response) {
                android.util.Log.d("ChatRepository", "sendMessage response code: " + response.code());
                android.util.Log.d("ChatRepository", "sendMessage response body: " + response.body());

                if (response.isSuccessful() && response.body() != null) {
                    android.util.Log.d("ChatRepository", "Response content: " + response.body().response);
                    liveData.setValue(Resource.success(response.body()));
                } else {
                    String errorBody = "";
                    try {
                        if (response.errorBody() != null) {
                            errorBody = response.errorBody().string();
                            android.util.Log.e("ChatRepository", "Error body: " + errorBody);
                        }
                    } catch (Exception e) {}
                    liveData.setValue(Resource.error("Send failed: HTTP " + response.code() + " - " + errorBody));
                }
            }

            @Override
            public void onFailure(Call<ChatResponse> call, Throwable t) {
                android.util.Log.e("ChatRepository", "sendMessage network error: " + t.getMessage());
                android.util.Log.e("ChatRepository", "Stack trace: ", t);
                liveData.setValue(Resource.error("Network error: " + t.getMessage()));
            }
        });
    }
}