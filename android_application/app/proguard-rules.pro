# Keep Gson models
-keep class com.example.multillm_langraphchatbot.data.model.** { *; }
-keep class com.example.multillm_langraphchatbot.network.** { *; }

# Keep Retrofit
-keepattributes Signature, InnerClasses, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-dontwarn org.codehaus.mojo.animal_sniffer.IgnoreJRERequirement
-dontwarn javax.annotation.**

# Keep Gson
-keep class com.google.gson.** { *; }
-keep class com.google.gson.stream.** { *; }

# Keep OkHttp
-keep class okhttp3.** { *; }
-keep interface okhttp3.** { *; }
-dontwarn okhttp3.**
-dontwarn okio.**

# Keep ViewBinding
-keep class androidx.viewbinding.** { *; }