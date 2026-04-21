package com.example.multillm_langraphchatbot.ui.chat;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.multillm_langraphchatbot.R;
import com.example.multillm_langraphchatbot.databinding.FragmentChatBinding;
import com.example.multillm_langraphchatbot.ui.threads.ThreadListViewModel;
import com.example.multillm_langraphchatbot.util.AppPreferences;
import com.google.android.material.snackbar.Snackbar;

public class ChatFragment extends Fragment {

    private FragmentChatBinding binding;
    private ChatViewModel chatViewModel;
    private ThreadListViewModel threadListViewModel;
    private MessageAdapter adapter;
    private AppPreferences prefs;
    private String currentThreadId;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentChatBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        prefs = new AppPreferences(requireContext());

        // ViewModels
        chatViewModel = new ViewModelProvider(this).get(ChatViewModel.class);
        threadListViewModel = new ViewModelProvider(requireActivity()).get(ThreadListViewModel.class);

        // RecyclerView
        adapter = new MessageAdapter();
        LinearLayoutManager layoutManager = new LinearLayoutManager(requireContext());
        layoutManager.setStackFromEnd(true);
        binding.rvMessages.setLayoutManager(layoutManager);
        binding.rvMessages.setAdapter(adapter);

        // Send button
        binding.btnSend.setOnClickListener(v -> onSendClicked());

        // Enter key to send
        binding.etMessage.setOnEditorActionListener((v, actionId, event) -> {
            onSendClicked();
            return true;
        });

        // Observe active thread
        threadListViewModel.activeThreadId.observe(getViewLifecycleOwner(), threadId -> {
            if (threadId != null && !threadId.equals(currentThreadId)) {
                currentThreadId = threadId;
                prefs.setLastThreadId(threadId);
                chatViewModel.loadMessages(threadId);
            }
        });

        // Observe messages
        chatViewModel.messages.observe(getViewLifecycleOwner(), resource -> {
            if (resource == null) return;

            binding.progressBar.setVisibility(resource.isLoading() ? View.VISIBLE : View.GONE);

            if (resource.isSuccess() && resource.data != null) {
                adapter.submitList(resource.data);
                if (!resource.data.isEmpty()) {
                    binding.rvMessages.smoothScrollToPosition(resource.data.size() - 1);
                    binding.tvEmpty.setVisibility(View.GONE);
                } else {
                    binding.tvEmpty.setVisibility(View.VISIBLE);
                }
            } else if (resource.isError()) {
                Snackbar.make(binding.getRoot(), resource.message, Snackbar.LENGTH_LONG)
                        .setAction(R.string.retry, v -> chatViewModel.loadMessages(currentThreadId))
                        .show();
            }
        });

        // Observe sending state (typing indicator)
        chatViewModel.isSending.observe(getViewLifecycleOwner(), sending -> {
            binding.typingIndicator.setVisibility(sending ? View.VISIBLE : View.GONE);
            binding.btnSend.setEnabled(!sending);
            binding.etMessage.setEnabled(!sending);

            // Auto-scroll when typing starts
            if (sending && adapter.getItemCount() > 0) {
                binding.rvMessages.smoothScrollToPosition(adapter.getItemCount() - 1);
            }
        });

        // CRITICAL: Observe sendResult to get the assistant's reply
        chatViewModel.sendResult.observe(getViewLifecycleOwner(), resource -> {
            if (resource == null) return;

            if (resource.isSuccess() && resource.data != null) {
                // Append the assistant's reply to the chat
                chatViewModel.appendAssistantMessage(resource.data.response);

                // Refresh thread list (auto-title may have updated)
                threadListViewModel.loadThreads();

            } else if (resource.isError()) {
                // Remove the optimistic user message? Or just show error
                Snackbar.make(binding.getRoot(),
                        getString(R.string.send_error, resource.message),
                        Snackbar.LENGTH_LONG).show();
                // Reset sending state
                chatViewModel.isSending.setValue(false);
            }
        });

        // Restore last thread
        if (threadListViewModel.activeThreadId.getValue() == null) {
            String lastThread = prefs.getLastThreadId();
            if (lastThread != null) {
                threadListViewModel.selectThread(lastThread);
            }
        }
    }

    private void onSendClicked() {
        if (currentThreadId == null) {
            Snackbar.make(binding.getRoot(), "Select a conversation first", Snackbar.LENGTH_SHORT).show();
            return;
        }

        String text = binding.etMessage.getText().toString().trim();
        if (TextUtils.isEmpty(text)) return;

        // Clear input field
        binding.etMessage.setText("");

        // Get selected model
        String model = prefs.getSelectedModel();

        // Send message
        chatViewModel.sendMessage(currentThreadId, text, model);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}