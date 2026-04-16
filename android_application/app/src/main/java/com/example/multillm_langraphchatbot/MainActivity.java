package com.example.multillm_langraphchatbot;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.widget.NestedScrollView;
import com.example.multillm_langraphchatbot.network.ApiClient;
import com.example.multillm_langraphchatbot.network.ApiService;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    private EditText messageInput;
    private Button sendButton, newChatButton, testButton;
    private TextView chatDisplay, statusText;
    private ProgressBar progressBar;
    private NestedScrollView scrollView;

    private ApiService apiService;
    private String currentThreadId = null;
    private final StringBuilder conversationHistory = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize views
        messageInput = findViewById(R.id.etMessage);
        sendButton = findViewById(R.id.btnSend);
        newChatButton = findViewById(R.id.btnNewChat);
        testButton = findViewById(R.id.btnTestConnection);
        chatDisplay = findViewById(R.id.tvChatHistory);
        statusText = findViewById(R.id.tvStatus);
        progressBar = findViewById(R.id.progressBar);
        scrollView = findViewById(R.id.scrollView);

        // Initialize API
        apiService = ApiClient.getClient().create(ApiService.class);

        // Setup click listeners
        testButton.setOnClickListener(v -> testConnection());
        newChatButton.setOnClickListener(v -> createNewThread());
        sendButton.setOnClickListener(v -> sendMessage());

        statusText.setText("Ready. Tap 'Test' to connect.");
    }

    private void testConnection() {
        setLoading(true);
        appendToChat("🔄 Testing connection...\n");

        apiService.checkHealth().enqueue(new Callback<ApiService.HealthResponse>() {
            @Override
            public void onResponse(@NonNull Call<ApiService.HealthResponse> call,
                                   @NonNull Response<ApiService.HealthResponse> response) {
                setLoading(false);
                if (response.isSuccessful() && response.body() != null) {
                    appendToChat("✅ Connected!\n");
                    statusText.setText("Connected ✓");
                    Toast.makeText(MainActivity.this, "Connected!", Toast.LENGTH_SHORT).show();
                    if (currentThreadId == null) createNewThread();
                } else {
                    appendToChat("❌ Connection failed (HTTP " + response.code() + ")\n");
                    statusText.setText("Connection failed");
                }
            }

            @Override
            public void onFailure(@NonNull Call<ApiService.HealthResponse> call, @NonNull Throwable t) {
                setLoading(false);
                appendToChat("❌ Error: " + t.getMessage() + "\n");
                appendToChat("💡 Make sure backend is running on port 8000\n");
                statusText.setText("Error: " + t.getMessage());
                Toast.makeText(MainActivity.this, "Connection error!", Toast.LENGTH_LONG).show();
            }
        });
    }

    private void createNewThread() {
        setLoading(true);
        appendToChat("🔄 Creating new chat...\n");

        apiService.createThread("Mobile Chat", "llama-8b-instant").enqueue(new Callback<ApiService.ThreadResponse>() {
            @Override
            public void onResponse(@NonNull Call<ApiService.ThreadResponse> call,
                                   @NonNull Response<ApiService.ThreadResponse> response) {
                setLoading(false);
                if (response.isSuccessful() && response.body() != null) {
                    currentThreadId = response.body().thread_id;
                    appendToChat("✅ New chat created!\n\n");
                    statusText.setText("Ready to chat");
                    enableChatInput(true);
                    Toast.makeText(MainActivity.this, "Chat ready!", Toast.LENGTH_SHORT).show();
                    conversationHistory.setLength(0);
                    chatDisplay.setText(conversationHistory.toString());
                } else {
                    appendToChat("❌ Failed to create chat\n");
                }
            }

            @Override
            public void onFailure(@NonNull Call<ApiService.ThreadResponse> call, @NonNull Throwable t) {
                setLoading(false);
                appendToChat("❌ Error: " + t.getMessage() + "\n");
            }
        });
    }

    private void sendMessage() {
        String message = messageInput.getText().toString().trim();
        if (message.isEmpty()) {
            Toast.makeText(this, "Enter a message", Toast.LENGTH_SHORT).show();
            return;
        }

        if (currentThreadId == null) {
            Toast.makeText(this, "Create a new chat first", Toast.LENGTH_SHORT).show();
            createNewThread();
            return;
        }

        setLoading(true);
        appendToChat("👤 You: " + message + "\n");
        appendToChat("🤖 Assistant: ");
        messageInput.setText("");

        ApiService.ChatRequest request = new ApiService.ChatRequest(currentThreadId, message, "llama-8b-instant");
        apiService.sendMessage(request).enqueue(new Callback<ApiService.ChatResponse>() {
            @Override
            public void onResponse(@NonNull Call<ApiService.ChatResponse> call,
                                   @NonNull Response<ApiService.ChatResponse> response) {
                setLoading(false);
                if (response.isSuccessful() && response.body() != null) {
                    appendToChat(response.body().response + "\n\n");
                } else {
                    appendToChat("[Error: " + response.code() + "]\n\n");
                }
            }

            @Override
            public void onFailure(@NonNull Call<ApiService.ChatResponse> call, @NonNull Throwable t) {
                setLoading(false);
                appendToChat("[Error: " + t.getMessage() + "]\n\n");
            }
        });
    }

    private void appendToChat(String text) {
        conversationHistory.append(text);
        runOnUiThread(() -> {
            chatDisplay.setText(conversationHistory.toString());
            scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
        });
    }

    private void enableChatInput(boolean enable) {
        runOnUiThread(() -> {
            messageInput.setEnabled(enable);
            sendButton.setEnabled(enable);
        });
    }

    private void setLoading(boolean loading) {
        runOnUiThread(() -> {
            progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            testButton.setEnabled(!loading);
            newChatButton.setEnabled(!loading);
            if (!loading && currentThreadId != null) {
                sendButton.setEnabled(true);
                messageInput.setEnabled(true);
            }
        });
    }
}