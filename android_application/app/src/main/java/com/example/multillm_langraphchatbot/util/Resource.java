package com.example.multillm_langraphchatbot.util;

/**
 * Wrapper that represents a UI state: Loading, Success, or Error.
 * Used by LiveData to communicate repository results to the UI.
 */
public class Resource<T> {

    public enum Status { LOADING, SUCCESS, ERROR }

    public final Status status;
    public final T data;
    public final String message;

    private Resource(Status status, T data, String message) {
        this.status  = status;
        this.data    = data;
        this.message = message;
    }

    public static <T> Resource<T> loading() {
        return new Resource<>(Status.LOADING, null, null);
    }

    public static <T> Resource<T> success(T data) {
        return new Resource<>(Status.SUCCESS, data, null);
    }

    public static <T> Resource<T> error(String message) {
        return new Resource<>(Status.ERROR, null, message);
    }

    public boolean isLoading()  { return status == Status.LOADING; }
    public boolean isSuccess()  { return status == Status.SUCCESS; }
    public boolean isError()    { return status == Status.ERROR; }
}