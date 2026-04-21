package com.example.multillm_langraphchatbot.ui.chat;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.example.multillm_langraphchatbot.R;
import com.example.multillm_langraphchatbot.data.model.Message;

/**
 * RecyclerView adapter for chat messages with two view types:
 *   VIEW_TYPE_USER      → item_message_user.xml   (right-aligned bubble)
 *   VIEW_TYPE_ASSISTANT → item_message_assistant.xml (left-aligned bubble)
 */
public class MessageAdapter extends ListAdapter<Message, RecyclerView.ViewHolder> {

    private static final int VIEW_TYPE_USER = 0;
    private static final int VIEW_TYPE_ASSISTANT = 1;

    public MessageAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<Message> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<Message>() {
                @Override
                public boolean areItemsTheSame(@NonNull Message a, @NonNull Message b) {
                    return a == b;
                }

                @Override
                public boolean areContentsTheSame(@NonNull Message a, @NonNull Message b) {
                    return a.role.equals(b.role) && a.content.equals(b.content);
                }
            };

    @Override
    public int getItemViewType(int position) {
        return getItem(position).isUser() ? VIEW_TYPE_USER : VIEW_TYPE_ASSISTANT;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        if (viewType == VIEW_TYPE_USER) {
            View view = inflater.inflate(R.layout.item_message_user, parent, false);
            return new UserViewHolder(view);
        } else {
            View view = inflater.inflate(R.layout.item_message_assistant, parent, false);
            return new AssistantViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        Message msg = getItem(position);
        if (holder instanceof UserViewHolder) {
            ((UserViewHolder) holder).bind(msg);
        } else if (holder instanceof AssistantViewHolder) {
            ((AssistantViewHolder) holder).bind(msg);
        }
    }

    static class UserViewHolder extends RecyclerView.ViewHolder {
        final TextView tvContent;

        UserViewHolder(View itemView) {
            super(itemView);
            tvContent = itemView.findViewById(R.id.tvMessageContent);
        }

        void bind(Message msg) {
            tvContent.setText(msg.content);
        }
    }

    static class AssistantViewHolder extends RecyclerView.ViewHolder {
        final TextView tvContent;

        AssistantViewHolder(View itemView) {
            super(itemView);
            tvContent = itemView.findViewById(R.id.tvMessageContent);
        }

        void bind(Message msg) {
            tvContent.setText(msg.content);
        }
    }
}