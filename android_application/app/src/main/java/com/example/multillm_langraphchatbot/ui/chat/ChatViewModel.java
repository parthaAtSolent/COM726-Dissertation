package com.example.multillm_langraphchatbot.ui.chat;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.multillm_langraphchatbot.data.model.ChatResponse;
import com.example.multillm_langraphchatbot.data.model.Message;
import com.example.multillm_langraphchatbot.data.repository.ChatRepository;
import com.example.multillm_langraphchatbot.util.Resource;

import java.util.ArrayList;
import java.util.List;

public class ChatViewModel extends ViewModel {

    private final ChatRepository repository = new ChatRepository();

    public final MutableLiveData<Resource<List<Message>>> messages   = new MutableLiveData<>();
    public final MutableLiveData<Resource<ChatResponse>>  sendResult = new MutableLiveData<>();
    public final MutableLiveData<Boolean>                 isSending  = new MutableLiveData<>(false);

    private final List<Message> localMessages = new ArrayList<>();

    public void loadMessages(String threadId) {
        repository.loadMessages(threadId, messages);
    }


    public void sendMessage(String threadId, String text, String model) {
        // 1. Append user message optimistically
        Message userMsg = new Message();
        userMsg.role = "user";
        userMsg.content = text;
        localMessages.add(userMsg);
        pushMessages();

        // 2. Show typing indicator
        isSending.setValue(true);

        // 3. Send to server - result will go to sendResult LiveData
        repository.sendMessage(threadId, text, model, sendResult);
    }

    /** Called by ChatFragment when a successful reply arrives. */
    public void appendAssistantMessage(String content) {
        Message msg = new Message();
        msg.role    = "assistant";
        msg.content = content;
        localMessages.add(msg);
        isSending.setValue(false);
        pushMessages();
    }

    public void setMessages(List<Message> msgs) {
        localMessages.clear();
        if (msgs != null) localMessages.addAll(msgs);
        pushMessages();
    }

    private void pushMessages() {
        messages.setValue(Resource.success(new ArrayList<>(localMessages)));
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        localMessages.clear();
    }
}