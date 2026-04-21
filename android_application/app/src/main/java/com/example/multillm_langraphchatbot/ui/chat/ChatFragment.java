package com.example.multillm_langraphchatbot.ui.chat;

import android.app.Dialog;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.multillm_langraphchatbot.R;
import com.example.multillm_langraphchatbot.data.model.ModelOption;
import com.example.multillm_langraphchatbot.databinding.FragmentChatBinding;
import com.example.multillm_langraphchatbot.ui.threads.ThreadListViewModel;
import com.example.multillm_langraphchatbot.util.AppPreferences;
import com.example.multillm_langraphchatbot.util.ModelHelper;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.snackbar.Snackbar;

import java.util.List;

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

        // Setup model selector button
        setupModelSelector();

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

        // Observe sending state
        chatViewModel.isSending.observe(getViewLifecycleOwner(), sending -> {
            binding.typingIndicator.setVisibility(sending ? View.VISIBLE : View.GONE);
            binding.btnSend.setEnabled(!sending);
            binding.etMessage.setEnabled(!sending);
            binding.btnSelectModel.setEnabled(!sending);

            if (sending && adapter.getItemCount() > 0) {
                binding.rvMessages.smoothScrollToPosition(adapter.getItemCount() - 1);
            }
        });

        // Observe sendResult
        chatViewModel.sendResult.observe(getViewLifecycleOwner(), resource -> {
            if (resource == null) return;

            if (resource.isSuccess() && resource.data != null) {
                chatViewModel.appendAssistantMessage(resource.data.response);
                threadListViewModel.loadThreads();
            } else if (resource.isError()) {
                Snackbar.make(binding.getRoot(),
                        getString(R.string.send_error, resource.message),
                        Snackbar.LENGTH_LONG).show();
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

    private void setupModelSelector() {
        String currentModel = prefs.getSelectedModel();
        updateModelButtonText(currentModel);

        binding.btnSelectModel.setOnClickListener(v -> showModelSelectorDialog());
    }

    private void showModelSelectorDialog() {
        Dialog dialog = new Dialog(requireContext());
        dialog.setContentView(R.layout.dialog_model_selector);

        ListView listView = dialog.findViewById(R.id.listViewModels);
        List<ModelOption> models = ModelHelper.getAvailableModels();
        String currentModel = prefs.getSelectedModel();

        // Create custom adapter
        ArrayAdapter<ModelOption> adapter = new ArrayAdapter<ModelOption>(
                requireContext(),
                R.layout.item_model,
                R.id.tvModelName,
                models
        ) {
            @NonNull
            @Override
            public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
                View view = super.getView(position, convertView, parent);
                ModelOption model = getItem(position);

                TextView tvModelName = view.findViewById(R.id.tvModelName);
                TextView tvModelDesc = view.findViewById(R.id.tvModelDescription);

                tvModelName.setText(model.getDisplayName());
                tvModelDesc.setText(model.getDescription());

                // Highlight current selection
                if (model.getKey().equals(currentModel)) {
                    view.setBackgroundColor(requireContext().getColor(R.color.primary_container));
                } else {
                    view.setBackgroundColor(0);
                }

                return view;
            }
        };

        listView.setAdapter(adapter);

        listView.setOnItemClickListener((parent, view, position, id) -> {
            ModelOption selected = models.get(position);
            prefs.setSelectedModel(selected.getKey());
            updateModelButtonText(selected.getKey());
            dialog.dismiss();

            Snackbar.make(binding.getRoot(),
                    "Switched to " + selected.getDisplayName(),
                    Snackbar.LENGTH_SHORT).show();
        });

        dialog.show();
    }

    private void updateModelButtonText(String modelKey) {
        List<ModelOption> models = ModelHelper.getAvailableModels();
        for (ModelOption model : models) {
            if (model.getKey().equals(modelKey)) {
                binding.btnSelectModel.setText(model.getDisplayName());
                return;
            }
        }
        binding.btnSelectModel.setText("🤖 Custom");
    }

    private void onSendClicked() {
        if (currentThreadId == null) {
            Snackbar.make(binding.getRoot(), "Select a conversation first", Snackbar.LENGTH_SHORT).show();
            return;
        }

        String text = binding.etMessage.getText().toString().trim();
        if (TextUtils.isEmpty(text)) return;

        binding.etMessage.setText("");

        // Get selected model (custom or specific)
        String modelKey = prefs.getSelectedModel();
        String apiModel = ModelHelper.getModelKeyForApi(modelKey);

        chatViewModel.sendMessage(currentThreadId, text, apiModel);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}