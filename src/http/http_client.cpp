/**
 * HTTP Client Implementation
 *
 * Uses libcurl for HTTP/HTTPS requests with gzip decompression support.
 * On iOS, this is a stub - HTTP requests should use NSURLSession (TODO).
 */

#include "mystral/http/http_client.h"

#if defined(MYSTRAL_HTTP_FOUNDATION) || defined(MYSTRAL_HTTP_ANDROID)
// iOS/Android stub implementation - HTTP not yet implemented
// iOS: TODO: Implement using NSURLSession in Objective-C++
// Android: TODO: Implement using HttpURLConnection via JNI

#include <iostream>
#ifdef MYSTRAL_HTTP_ANDROID
#include <SDL3/SDL_system.h>
#include <jni.h>
#include <algorithm>
#include <limits>
#endif

namespace mystral {
namespace http {

HttpClient::HttpClient() {}

HttpClient::~HttpClient() {}

HttpResponse HttpClient::get(const std::string& url, const HttpOptions& options) {
    return request("GET", url, {}, options);
}

HttpResponse HttpClient::post(const std::string& url, const std::vector<uint8_t>& body, const HttpOptions& options) {
    return request("POST", url, body, options);
}

HttpResponse HttpClient::request(const std::string& method, const std::string& url,
                                   const std::vector<uint8_t>& body, const HttpOptions& options) {
    HttpResponse response;
    response.ok = false;
    response.status = 0;
#ifdef MYSTRAL_HTTP_ANDROID
    response.url = url;
    if (!options.verifySSL) {
        response.error = "Android HTTP requires certificate verification";
        return response;
    }
    auto* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    if (!env || env->PushLocalFrame(32) < 0) {
        response.error = "Android HTTP JNI environment is unavailable";
        return response;
    }
    struct LocalFrame { JNIEnv* env; ~LocalFrame() { env->PopLocalFrame(nullptr); } } frame{env};
    auto activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!activity) { response.error = "Android HTTP Activity is unavailable"; return response; }
    auto activityClass = env->GetObjectClass(activity);
    auto request = env->GetStaticMethodID(activityClass, "performHttpRequest",
        "(Ljava/lang/String;Ljava/lang/String;[B[Ljava/lang/String;I)Lcom/mystral/engine/MystralActivity$HttpResult;");
    if (env->ExceptionCheck() || !request) {
        env->ExceptionClear(); response.error = "Android HTTP bridge method is unavailable"; return response;
    }
    if (body.size() > static_cast<size_t>(std::numeric_limits<jsize>::max()) ||
        options.headers.size() > static_cast<size_t>(std::numeric_limits<jsize>::max() / 2)) {
        response.error = "Android HTTP request is too large"; return response;
    }
    auto bodyArray = env->NewByteArray(static_cast<jsize>(body.size()));
    if (!body.empty()) env->SetByteArrayRegion(bodyArray, 0, static_cast<jsize>(body.size()), reinterpret_cast<const jbyte*>(body.data()));
    auto stringClass = env->FindClass("java/lang/String");
    auto headers = env->NewObjectArray(static_cast<jsize>(options.headers.size() * 2), stringClass, nullptr);
    jsize index = 0;
    for (const auto& header : options.headers) {
        auto key = env->NewStringUTF(header.first.c_str());
        auto value = env->NewStringUTF(header.second.c_str());
        env->SetObjectArrayElement(headers, index++, key);
        env->SetObjectArrayElement(headers, index++, value);
        env->DeleteLocalRef(key); env->DeleteLocalRef(value);
    }
    const auto seconds = std::min<long>(options.timeout > 0 ? options.timeout : 30, 3600);
    auto result = env->CallStaticObjectMethod(activityClass, request, env->NewStringUTF(method.c_str()),
        env->NewStringUTF(url.c_str()), bodyArray, headers, static_cast<jint>(seconds * 1000));
    if (env->ExceptionCheck() || !result) {
        env->ExceptionClear(); response.error = "Android HTTP request failed in Java"; return response;
    }
    auto resultClass = env->GetObjectClass(result);
    auto statusField = env->GetFieldID(resultClass, "status", "I");
    auto errorField = env->GetFieldID(resultClass, "error", "Ljava/lang/String;");
    auto urlField = env->GetFieldID(resultClass, "url", "Ljava/lang/String;");
    auto headersField = env->GetFieldID(resultClass, "headers", "[Ljava/lang/String;");
    auto bodyField = env->GetFieldID(resultClass, "body", "[B");
    if (env->ExceptionCheck() || !statusField || !errorField || !urlField || !headersField || !bodyField) {
        env->ExceptionClear(); response.error = "Android HTTP response fields are unavailable"; return response;
    }
    auto text = [env](jstring value) {
        std::string result;
        if (value) {
            const char* utf = env->GetStringUTFChars(value, nullptr);
            if (utf) { result = utf; env->ReleaseStringUTFChars(value, utf); }
        }
        return result;
    };
    response.status = env->GetIntField(result, statusField);
    response.error = text(static_cast<jstring>(env->GetObjectField(result, errorField)));
    auto actualUrl = text(static_cast<jstring>(env->GetObjectField(result, urlField)));
    if (!actualUrl.empty()) response.url = actualUrl;
    auto output = static_cast<jbyteArray>(env->GetObjectField(result, bodyField));
    if (output) {
        const auto length = env->GetArrayLength(output);
        response.data.resize(static_cast<size_t>(length));
        if (length) env->GetByteArrayRegion(output, 0, length, reinterpret_cast<jbyte*>(response.data.data()));
    }
    auto pairs = static_cast<jobjectArray>(env->GetObjectField(result, headersField));
    if (pairs) for (jsize i = 0; i + 1 < env->GetArrayLength(pairs); i += 2) {
        auto key = static_cast<jstring>(env->GetObjectArrayElement(pairs, i));
        auto value = static_cast<jstring>(env->GetObjectArrayElement(pairs, i + 1));
        response.headers[text(key)] = text(value);
        env->DeleteLocalRef(key); env->DeleteLocalRef(value);
    }
    if (env->ExceptionCheck()) {
        env->ExceptionClear(); response.error = "Android HTTP response conversion failed";
    }
    response.ok = response.error.empty() && response.status >= 200 && response.status < 300;
    return response;
#else
    response.error = "HTTP client not implemented on iOS yet";
#endif
    std::cerr << "[HTTP] " << response.error << std::endl;
    return response;
}

// Global instance
HttpClient& getHttpClient() {
    static HttpClient client;
    return client;
}

} // namespace http
} // namespace mystral

#else
// Desktop implementation using libcurl

#include <curl/curl.h>
#include <iostream>
#include <cstring>

namespace mystral {
namespace http {

// CURL write callback - accumulates data into a vector
static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realSize = size * nmemb;
    auto* buffer = static_cast<std::vector<uint8_t>*>(userp);

    const uint8_t* data = static_cast<const uint8_t*>(contents);
    buffer->insert(buffer->end(), data, data + realSize);

    return realSize;
}

// Header callback to capture response headers
static size_t headerCallback(char* buffer, size_t size, size_t nitems, void* userdata) {
    size_t realSize = size * nitems;
    auto* headers = static_cast<std::map<std::string, std::string>*>(userdata);

    std::string line(buffer, realSize);

    // Parse header line
    size_t colonPos = line.find(':');
    if (colonPos != std::string::npos) {
        std::string key = line.substr(0, colonPos);
        std::string value = line.substr(colonPos + 1);

        // Trim whitespace
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
        while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) value.pop_back();

        // Convert key to lowercase for consistent access
        for (char& c : key) c = std::tolower(c);

        (*headers)[key] = value;
    }

    return realSize;
}

HttpClient::HttpClient() {
    // Initialize CURL globally (thread-safe in modern curl)
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

HttpClient::~HttpClient() {
    curl_global_cleanup();
}

HttpResponse HttpClient::get(const std::string& url, const HttpOptions& options) {
    return request("GET", url, {}, options);
}

HttpResponse HttpClient::post(const std::string& url, const std::vector<uint8_t>& body, const HttpOptions& options) {
    return request("POST", url, body, options);
}

HttpResponse HttpClient::request(const std::string& method, const std::string& url,
                                   const std::vector<uint8_t>& body, const HttpOptions& options) {
    HttpResponse response;
    response.ok = false;
    response.status = 0;

    CURL* curl = curl_easy_init();
    if (!curl) {
        response.error = "Failed to initialize CURL";
        return response;
    }

    // Set URL
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    // Set method
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (!body.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.data());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
        }
    } else if (method == "PUT") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        if (!body.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.data());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
        }
    } else if (method == "DELETE") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }
    // GET is the default

    // Set write callback
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.data);

    // Set header callback
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, headerCallback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &response.headers);

    // Enable automatic decompression (gzip, deflate, br)
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");

    // Follow redirects
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);

    // Set timeout
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, options.timeout > 0 ? options.timeout : 30L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    // SSL verification (can be disabled for testing)
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, options.verifySSL ? 1L : 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, options.verifySSL ? 2L : 0L);

    // Set custom headers
    struct curl_slist* headerList = nullptr;
    for (const auto& [key, value] : options.headers) {
        std::string header = key + ": " + value;
        headerList = curl_slist_append(headerList, header.c_str());
    }
    if (headerList) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList);
    }

    // User agent
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "MystralRuntime/0.1");

    // Perform request
    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        response.status = static_cast<int>(httpCode);
        response.ok = (httpCode >= 200 && httpCode < 300);

        // Get final URL (after redirects)
        char* finalUrl = nullptr;
        curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &finalUrl);
        if (finalUrl) {
            response.url = finalUrl;
        }
    } else {
        response.error = curl_easy_strerror(res);
        std::cerr << "[HTTP] Request failed: " << response.error << std::endl;
    }

    // Cleanup
    if (headerList) {
        curl_slist_free_all(headerList);
    }
    curl_easy_cleanup(curl);

    return response;
}

// Global instance
static std::unique_ptr<HttpClient> g_httpClient;

HttpClient& getHttpClient() {
    if (!g_httpClient) {
        g_httpClient = std::make_unique<HttpClient>();
    }
    return *g_httpClient;
}

} // namespace http
} // namespace mystral

#endif // MYSTRAL_HTTP_FOUNDATION
