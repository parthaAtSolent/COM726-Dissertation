plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.multillm_langraphchatbot"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.multillm_langraphchatbot"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Configurable server URL — override in local.properties for real device
        buildConfigField("String", "SERVER_BASE_URL", "\"${project.findProperty("serverBaseUrl") ?: "http://10.0.2.2:8000/"}\"")
    }

    buildTypes {
        debug {
            isDebuggable = true
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }
        release {
            isMinifyEnabled = false  // Temporarily disable minification
            isShrinkResources = false  // Temporarily disable shrinking
            manifestPlaceholders["usesCleartextTraffic"] = "true"  // Allow HTTP for testing
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
}

dependencies {
    // AndroidX Core
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Material Design 3
    implementation("com.google.android.material:material:1.12.0")

    // ViewModel + LiveData
    implementation("androidx.lifecycle:lifecycle-viewmodel:2.8.1")
    implementation("androidx.lifecycle:lifecycle-livedata:2.8.1")
    implementation("androidx.lifecycle:lifecycle-runtime:2.8.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-savedstate:2.8.1")

    // Fragment + Navigation
    implementation("androidx.fragment:fragment:1.7.1")
    implementation("androidx.navigation:navigation-fragment:2.7.7")
    implementation("androidx.navigation:navigation-ui:2.7.7")

    // RecyclerView
    implementation("androidx.recyclerview:recyclerview:1.3.2")

    // SwipeRefreshLayout
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.1.0")

    // DataStore (Preferences — replaces SharedPreferences)
    implementation("androidx.datastore:datastore-preferences:1.1.1")
    implementation("androidx.datastore:datastore-rxjava2:1.1.1")

    // Networking
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // JSON
    implementation("com.google.code.gson:gson:2.10.1")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}