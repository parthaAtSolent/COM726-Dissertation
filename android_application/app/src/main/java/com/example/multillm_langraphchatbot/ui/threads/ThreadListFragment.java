package com.example.multillm_langraphchatbot.ui.threads;

import android.app.AlertDialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.multillm_langraphchatbot.R;
import com.example.multillm_langraphchatbot.data.model.Thread;
import com.example.multillm_langraphchatbot.databinding.FragmentThreadListBinding;
import com.example.multillm_langraphchatbot.util.AppPreferences;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.snackbar.Snackbar;

import java.util.List;

/**
 * Thread list shown inside the NavigationDrawer.
 * Scoped to the Activity so it shares ThreadListViewModel with ChatFragment.
 */
public class ThreadListFragment extends Fragment implements ThreadAdapter.OnThreadClickListener {

    private FragmentThreadListBinding binding;
    private ThreadListViewModel       viewModel;
    private ThreadAdapter             adapter;
    private AppPreferences            prefs;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentThreadListBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        prefs     = new AppPreferences(requireContext());
        viewModel = new ViewModelProvider(requireActivity()).get(ThreadListViewModel.class);

        // RecyclerView
        adapter = new ThreadAdapter(this);
        binding.rvThreads.setLayoutManager(new LinearLayoutManager(requireContext()));
        binding.rvThreads.setAdapter(adapter);

        // FAB — New Chat
        binding.fabNewChat.setOnClickListener(v -> showNewChatDialog());

        // Observe thread list
        viewModel.threads.observe(getViewLifecycleOwner(), resource -> {
            binding.progressBar.setVisibility(resource.isLoading() ? View.VISIBLE : View.GONE);
            if (resource.isSuccess() && resource.data != null) {
                List<Thread> threads = resource.data;
                adapter.submitList(threads);
                binding.tvEmpty.setVisibility(threads.isEmpty() ? View.VISIBLE : View.GONE);
                // If no thread selected, select the most recent one
                if (viewModel.activeThreadId.getValue() == null && !threads.isEmpty()) {
                    viewModel.selectThread(threads.get(0).threadId);
                }
            } else if (resource.isError()) {
                Snackbar.make(binding.getRoot(), resource.message, Snackbar.LENGTH_LONG)
                        .setAction(R.string.retry, v -> viewModel.loadThreads())
                        .show();
            }
        });

        // Observe new thread creation
        viewModel.newThread.observe(getViewLifecycleOwner(), resource -> {
            if (resource.isSuccess() && resource.data != null) {
                viewModel.selectThread(resource.data.threadId);
                viewModel.loadThreads();
                // Close drawer after selecting new thread
                closeDrawer();
            } else if (resource.isError()) {
                Snackbar.make(binding.getRoot(), resource.message, Snackbar.LENGTH_LONG).show();
            }
        });

        // Observe active thread ID to highlight correct row
        viewModel.activeThreadId.observe(getViewLifecycleOwner(), threadId -> {
            adapter.setActiveThreadId(threadId);
        });

        // Observe delete result
        viewModel.deleteResult.observe(getViewLifecycleOwner(), resource -> {
            if (resource != null && resource.isSuccess()) {
                viewModel.loadThreads();
                // If deleted thread was active, clear or select next
                viewModel.activeThreadId.setValue(null);
            }
        });
    }

    @Override
    public void onThreadClick(Thread thread) {
        viewModel.selectThread(thread.threadId);
        closeDrawer();
    }

    @Override
    public void onThreadLongClick(Thread thread) {
        // Show context menu: Rename / Delete
        String[] options = {
                getString(R.string.rename),
                getString(R.string.delete)
        };
        new MaterialAlertDialogBuilder(requireContext())
                .setTitle(thread.title)
                .setItems(options, (dialog, which) -> {
                    if (which == 0) showRenameDialog(thread);
                    else            showDeleteConfirmation(thread);
                })
                .show();
    }

    private void showNewChatDialog() {
        String model = prefs.getSelectedModel();
        viewModel.createThread("New Chat", model);
    }

    private void showRenameDialog(Thread thread) {
        EditText input = new EditText(requireContext());
        input.setText(thread.title);
        input.setSelectAllOnFocus(true);

        new MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.rename_thread)
                .setView(input)
                .setPositiveButton(R.string.save, (d, w) -> {
                    String newTitle = input.getText().toString().trim();
                    if (!newTitle.isEmpty()) {
                        viewModel.renameThread(thread.threadId, newTitle);
                        viewModel.loadThreads();
                    }
                })
                .setNegativeButton(android.R.string.cancel, null)
                .show();
    }

    private void showDeleteConfirmation(Thread thread) {
        new MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.delete_thread_title)
                .setMessage(getString(R.string.delete_thread_message, thread.title))
                .setPositiveButton(R.string.delete, (d, w) -> viewModel.deleteThread(thread.threadId))
                .setNegativeButton(android.R.string.cancel, null)
                .show();
    }

    private void closeDrawer() {
        DrawerLayout drawer = requireActivity().findViewById(R.id.drawerLayout);
        if (drawer != null) drawer.closeDrawers();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}