package com.example.multillm_langraphchatbot.ui;

import android.os.Bundle;
import android.view.MenuItem;

import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.NavigationUI;

import com.example.multillm_langraphchatbot.R;
import com.example.multillm_langraphchatbot.databinding.ActivityMainBinding;
import com.example.multillm_langraphchatbot.ui.threads.ThreadListViewModel;

/**
 * Single-activity host.
 *
 * Layout:  DrawerLayout
 *           ├── NavHostFragment  (main content)
 *           └── NavigationView   (left drawer — thread list)
 *
 * The ThreadListViewModel is scoped to this Activity so that both
 * ChatFragment and ThreadListFragment share the same activeThreadId.
 */
public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private ThreadListViewModel threadListViewModel;
    private ActionBarDrawerToggle drawerToggle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        // ── Navigation ────────────────────────────────────────────────────────
        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.navHostFragment);
        NavController navController = navHostFragment.getNavController();
        NavigationUI.setupWithNavController(binding.bottomNav, navController);

        // ── Drawer toggle ──────────────────────────────────────────────────────
        drawerToggle = new ActionBarDrawerToggle(
                this, binding.drawerLayout,
                binding.toolbar,
                R.string.drawer_open,
                R.string.drawer_close);
        binding.drawerLayout.addDrawerListener(drawerToggle);
        drawerToggle.syncState();

        // ── Shared ViewModel ───────────────────────────────────────────────────
        threadListViewModel = new ViewModelProvider(this).get(ThreadListViewModel.class);

        // Load threads when the app starts
        threadListViewModel.loadThreads();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (drawerToggle.onOptionsItemSelected(item)) return true;
        return super.onOptionsItemSelected(item);
    }
}