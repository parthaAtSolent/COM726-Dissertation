package com.example.multillm_langraphchatbot.ui.rag;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.multillm_langraphchatbot.data.model.FilesResponse;
import com.example.multillm_langraphchatbot.data.model.UploadResponse;
import com.example.multillm_langraphchatbot.network.ApiClient;
import com.example.multillm_langraphchatbot.network.ApiService;
import com.example.multillm_langraphchatbot.util.Resource;

import java.io.File;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class RagViewModel extends ViewModel {

    private final ApiService api = ApiClient.getClient().create(ApiService.class);

    public final MutableLiveData<Resource<List<String>>>    files        = new MutableLiveData<>();
    public final MutableLiveData<Resource<UploadResponse>>  uploadResult = new MutableLiveData<>();

    public void loadFiles() {
        files.setValue(Resource.loading());
        api.getFiles().enqueue(new Callback<FilesResponse>() {
            @Override
            public void onResponse(Call<FilesResponse> call, Response<FilesResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    files.setValue(Resource.success(response.body().files));
                } else {
                    files.setValue(Resource.error("HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<FilesResponse> call, Throwable t) {
                files.setValue(Resource.error(t.getMessage()));
            }
        });
    }

    public void uploadFile(File file, String mimeType) {
        uploadResult.setValue(Resource.loading());
        RequestBody     reqBody  = RequestBody.create(MediaType.parse(mimeType), file);
        MultipartBody.Part part  = MultipartBody.Part.createFormData("file", file.getName(), reqBody);

        api.uploadFile(part).enqueue(new Callback<UploadResponse>() {
            @Override
            public void onResponse(Call<UploadResponse> call, Response<UploadResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    uploadResult.setValue(Resource.success(response.body()));
                    loadFiles(); // Refresh file list
                } else {
                    uploadResult.setValue(Resource.error("Upload failed: HTTP " + response.code()));
                }
            }

            @Override
            public void onFailure(Call<UploadResponse> call, Throwable t) {
                uploadResult.setValue(Resource.error(t.getMessage()));
            }
        });
    }
}