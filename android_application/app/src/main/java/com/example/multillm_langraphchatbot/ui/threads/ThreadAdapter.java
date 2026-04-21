package com.example.multillm_langraphchatbot.ui.threads;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.example.multillm_langraphchatbot.R;
import com.example.multillm_langraphchatbot.data.model.Thread;

public class ThreadAdapter extends ListAdapter<Thread, ThreadAdapter.ThreadViewHolder> {

    public interface OnThreadClickListener {
        void onThreadClick(Thread thread);
        void onThreadLongClick(Thread thread);
    }

    private final OnThreadClickListener listener;
    private String activeThreadId = null;

    public ThreadAdapter(OnThreadClickListener listener) {
        super(DIFF_CALLBACK);
        this.listener = listener;
    }

    public void setActiveThreadId(String id) {
        this.activeThreadId = id;
        notifyDataSetChanged(); // acceptable cost given small list size
    }

    private static final DiffUtil.ItemCallback<Thread> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<Thread>() {
                @Override
                public boolean areItemsTheSame(@NonNull Thread a, @NonNull Thread b) {
                    return a.threadId.equals(b.threadId);
                }

                @Override
                public boolean areContentsTheSame(@NonNull Thread a, @NonNull Thread b) {
                    return a.title.equals(b.title) && a.model.equals(b.model);
                }
            };

    @NonNull
    @Override
    public ThreadViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_thread, parent, false);
        return new ThreadViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ThreadViewHolder holder, int position) {
        Thread thread = getItem(position);
        holder.bind(thread, thread.threadId.equals(activeThreadId));
        holder.itemView.setOnClickListener(v -> listener.onThreadClick(thread));
        holder.itemView.setOnLongClickListener(v -> {
            listener.onThreadLongClick(thread);
            return true;
        });
    }

    static class ThreadViewHolder extends RecyclerView.ViewHolder {
        final TextView tvTitle;
        final TextView tvModel;

        ThreadViewHolder(View itemView) {
            super(itemView);
            tvTitle = itemView.findViewById(R.id.tvThreadTitle);
            tvModel = itemView.findViewById(R.id.tvThreadModel);
        }

        void bind(Thread thread, boolean isActive) {
            tvTitle.setText(thread.title != null ? thread.title : "New Chat");
            tvModel.setText(thread.model != null ? thread.model : "");
            itemView.setSelected(isActive);
            tvTitle.setTypeface(null, isActive
                    ? android.graphics.Typeface.BOLD
                    : android.graphics.Typeface.NORMAL);
        }
    }
}