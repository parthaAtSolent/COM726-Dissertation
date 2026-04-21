package com.example.multillm_langraphchatbot.ui.threads;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.multillm_langraphchatbot.data.model.NewThreadResponse;
import com.example.multillm_langraphchatbot.data.model.Thread;
import com.example.multillm_langraphchatbot.data.repository.ThreadRepository;
import com.example.multillm_langraphchatbot.util.Resource;

import java.util.List;

public class ThreadListViewModel extends ViewModel {

    private final ThreadRepository repository = new ThreadRepository();

    public final MutableLiveData<Resource<List<Thread>>> threads        = new MutableLiveData<>();
    public final MutableLiveData<Resource<NewThreadResponse>> newThread = new MutableLiveData<>();
    public final MutableLiveData<Resource<Boolean>> deleteResult        = new MutableLiveData<>();
    public final MutableLiveData<Resource<Boolean>> renameResult        = new MutableLiveData<>();

    /** Selected thread ID — shared with ChatFragment via Activity scope. */
    public final MutableLiveData<String> activeThreadId = new MutableLiveData<>();

    public void loadThreads() {
        repository.fetchThreads(threads);
    }

    public void createThread(String title, String model) {
        repository.createThread(title, model, newThread);
    }

    public void deleteThread(String threadId) {
        repository.deleteThread(threadId, deleteResult);
    }

    public void renameThread(String threadId, String newTitle) {
        repository.renameThread(threadId, newTitle, renameResult);
    }

    public void selectThread(String threadId) {
        activeThreadId.setValue(threadId);
    }
}