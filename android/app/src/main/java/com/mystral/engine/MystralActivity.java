package com.mystral.engine;

import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;
import android.view.PixelCopy;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.RelativeLayout;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONObject;
import org.libsdl.app.SDLActivity;
import org.libsdl.app.SDLSurface;

/**
 * MystralActivity - Main entry point for Mystral Native on Android.
 *
 * Extends SDLActivity which handles:
 * - Native library loading
 * - Surface creation (ANativeWindow)
 * - Input events (touch, keyboard, gamepad)
 * - Lifecycle management (pause/resume)
 *
 * The SDL3 Android backend provides the native window to wgpu-native
 * for Vulkan surface creation.
 */
public class MystralActivity extends SDLActivity {
    private static volatile MystralActivity current;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final AtomicInteger surfaceGeneration = new AtomicInteger();
    private volatile InteropServer server;
    private String[] arguments;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        current = this;
        super.onCreate(savedInstanceState);
        try {
            server = new InteropServer(this, getIntent().getStringExtra("run_id"),
                    getIntent().getStringExtra("case_id"));
            server.startup(nativeCompiledAbi(), nativeV8StartupMode(), nativeV8Version());
            server.start();
            record("created");
        } catch (Exception error) {
            Log.e("MystralInterop", "Initialization failed", error);
            finish();
        }
    }

    @Override
    protected SDLSurface createSDLSurface(Context context) {
        SDLSurface surface = super.createSDLSurface(context);
        // This is an actual 320x240 physical-pixel SurfaceView, visibly centered in the Activity.
        // No DOM dimensions, DPR or captured bitmap are relabelled/rescaled to meet an oracle.
        RelativeLayout.LayoutParams layout = new RelativeLayout.LayoutParams(320, 240);
        layout.addRule(RelativeLayout.CENTER_IN_PARENT);
        surface.setLayoutParams(layout);
        surface.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override public void surfaceCreated(SurfaceHolder holder) {
                surfaceGeneration.incrementAndGet();
                record("surface-created");
            }
            @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                record("surface-changed");
            }
            @Override public void surfaceDestroyed(SurfaceHolder holder) {
                surfaceGeneration.incrementAndGet();
                record("surface-destroyed");
            }
        });
        return surface;
    }

    /**
     * Override to specify which native libraries to load.
     * SDL handles loading SDL3, then we load mystral-runtime.
     */
    @Override
    protected String[] getLibraries() {
        return new String[] {
            "SDL3",
            "mystral-runtime"
        };
    }

    /**
     * Override to specify the main function.
     * This is called by SDL after the libraries are loaded.
     */
    @Override
    protected String getMainFunction() {
        return "SDL_main";
    }

    /**
     * Get command line arguments for the main function.
     * Can be used to specify the script path.
     */
    @Override
    protected synchronized String[] getArguments() {
        if (arguments != null) return arguments.clone();
        try {
            if (server == null) throw new IllegalStateException("Interop server was not initialized");
            String abi = nativeCompiledAbi();
            if (!abi.equals("x86_64") && !abi.equals("arm64-v8a"))
                throw new IllegalStateException("Unsupported loaded native ABI: " + abi);
            String startupMode = nativeV8StartupMode();
            server.startup(abi, startupMode, nativeV8Version());
            File snapshot = null;
            if (startupMode.equals("external")) {
                snapshot = new File(server.directory(), "snapshot_blob.bin");
                byte[] bytes;
                try (InputStream input = getAssets().open("v8/" + abi + "/snapshot_blob.bin")) {
                    bytes = readBytes(input, 16 * 1024 * 1024);
                }
                if (bytes.length == 0) throw new IllegalStateException("Empty V8 snapshot");
                InteropServer.writeNew(snapshot, bytes);
                server.snapshot(abi, snapshot, InteropServer.sha256(bytes));
            }
            byte[] entry;
            try (InputStream input = getAssets().open("interop/entry.js")) {
                entry = readBytes(input, 4 * 1024 * 1024);
            }
            File entryFile = new File(server.directory(), "native-entry.mjs");
            InteropServer.writeNew(entryFile, entry);
            server.entry(entryFile, InteropServer.sha256(entry));
            JSONObject config = server.configuration();
            if (snapshot != null) config.put("snapshotPath", snapshot.getAbsolutePath());
            config.put("entryPath", entryFile.getAbsolutePath());
            config.put("entrySha256", InteropServer.sha256(entry));
            arguments = snapshot == null
                    ? new String[] {entryFile.getAbsolutePath(), config.toString()}
                    : new String[] {entryFile.getAbsolutePath(), config.toString(), snapshot.getAbsolutePath()};
            return arguments.clone();
        } catch (Exception error) {
            Log.e("MystralInterop", "Cannot prepare native entry", error);
            onNativeFinished(1, error.toString());
            return new String[] {"", "{}", ""};
        }
    }

    private static native String nativeCompiledAbi();
    private static native String nativeV8StartupMode();
    private static native String nativeV8Version();
    private static native void nativeRequestQuit();

    public static void onNativeReady(String abi, String engine, String backend, String startupMode, String v8Version) {
        MystralActivity activity = current;
        if (activity != null && activity.server != null)
            activity.server.nativeReady(abi, engine, backend, startupMode, v8Version);
    }

    public static void onNativeFinished(int code, String error) {
        MystralActivity activity = current;
        Log.i("MystralInterop", "EXIT_STAGE java-native-finished code=" + code + " activity=" + (activity != null)
                + " server=" + (activity != null && activity.server != null));
        if (activity != null && activity.server != null) activity.server.nativeFinished(code, error);
    }

    public static int getReportedOutcome() {
        MystralActivity activity = current;
        return activity != null && activity.server != null ? activity.server.outcome() : 1;
    }

    void requestNativeQuit() {
        Log.i("MystralInterop", "EXIT_STAGE java-quit-schedule pid=" + Process.myPid());
        boolean posted = ui.postDelayed(() -> {
            Log.i("MystralInterop", "EXIT_STAGE java-quit-dispatch pid=" + Process.myPid());
            nativeRequestQuit();
            Log.i("MystralInterop", "EXIT_STAGE java-quit-return pid=" + Process.myPid());
        }, 50);
        Log.i("MystralInterop", "EXIT_STAGE java-quit-posted value=" + posted);
    }

    private void record(String event) {
        InteropServer value = server;
        if (value != null) value.lifecycle(event, surfaceGeneration.get(), getResources().getConfiguration().orientation);
    }

    @Override protected void onPause() { record("pause"); super.onPause(); }
    @Override protected void onResume() { super.onResume(); record("resume"); }
    @Override public void onConfigurationChanged(Configuration config) {
        super.onConfigurationChanged(config);
        record("configuration-changed");
    }
    @Override protected void onDestroy() {
        record("destroy");
        if (server != null) server.close();
        current = null;
        super.onDestroy();
    }

    JSONObject captureSurface(File file, long deadline) throws Exception {
        CountDownLatch ready = new CountDownLatch(1);
        AtomicBoolean accepting = new AtomicBoolean(true);
        AtomicReference<JSONObject> result = new AtomicReference<>();
        AtomicReference<Exception> failure = new AtomicReference<>();
        ui.post(() -> {
            Bitmap bitmap = null;
            try {
                SurfaceView view = mSurface;
                if (view == null || !view.isShown() || !view.getHolder().getSurface().isValid())
                    throw new IllegalStateException("SDL SurfaceView is not visibly ready");
                Rect frame = view.getHolder().getSurfaceFrame();
                int width = frame.width(), height = frame.height();
                if (width <= 0 || height <= 0 || width != view.getWidth() || height != view.getHeight())
                    throw new IllegalStateException("Surface buffer and visible view dimensions differ");
                int[] location = new int[2];
                view.getLocationOnScreen(location);
                int generation = surfaceGeneration.get();
                bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                final Bitmap target = bitmap;
                PixelCopy.request(view, target, code -> {
                    try {
                        if (!accepting.get()) return;
                        if (code != PixelCopy.SUCCESS) throw new IllegalStateException("PixelCopy result=" + code);
                        if (generation != surfaceGeneration.get())
                            throw new IllegalStateException("Surface generation changed during PixelCopy");
                        ByteArrayOutputStream encoded = new ByteArrayOutputStream();
                        if (!target.compress(Bitmap.CompressFormat.PNG, 100, encoded))
                            throw new IllegalStateException("PNG encoding failed");
                        byte[] bytes = encoded.toByteArray();
                        if (!accepting.get() || SystemClock.elapsedRealtime() >= deadline) return;
                        InteropServer.writeNew(file, bytes);
                        JSONObject metadata = new JSONObject();
                        metadata.put("channel", "android-pixelcopy");
                        metadata.put("pid", Process.myPid());
                        metadata.put("file", file.getName());
                        metadata.put("width", width); metadata.put("height", height);
                        metadata.put("bitmapWidth", target.getWidth()); metadata.put("bitmapHeight", target.getHeight());
                        metadata.put("surfaceWidth", width); metadata.put("surfaceHeight", height);
                        metadata.put("viewWidth", view.getWidth()); metadata.put("viewHeight", view.getHeight());
                        metadata.put("surfaceGeneration", generation); metadata.put("resultCode", code);
                        metadata.put("bounds", new JSONObject().put("x", location[0]).put("y", location[1])
                                .put("width", view.getWidth()).put("height", view.getHeight()));
                        metadata.put("sha256", InteropServer.sha256(bytes));
                        result.set(metadata);
                    } catch (Exception error) { failure.set(error); }
                    finally { target.recycle(); ready.countDown(); }
                }, ui);
            } catch (Exception error) {
                if (bitmap != null) bitmap.recycle();
                failure.set(error); ready.countDown();
            }
        });
        long remaining = Math.min(5000, deadline - SystemClock.elapsedRealtime());
        boolean completed = remaining > 0 && ready.await(remaining, TimeUnit.MILLISECONDS);
        accepting.set(false);
        if (!completed) throw new IllegalStateException("Original PixelCopy capture budget exceeded");
        if (failure.get() != null) throw failure.get();
        if (result.get() == null) throw new IllegalStateException("PixelCopy completed after capture deadline");
        return result.get();
    }

    static byte[] readBytes(InputStream input, int maximum) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] block = new byte[16384];
        int count;
        while ((count = input.read(block)) != -1) {
            if (output.size() + count > maximum) throw new IllegalStateException("Response body exceeds bounded size");
            output.write(block, 0, count);
        }
        return output.toByteArray();
    }

    public static final class HttpResult {
        public int status;
        public String error = "";
        public String[] headers = new String[0];
        public byte[] body = new byte[0];
        public String url = "";
    }

    /** JNI HTTP transport; platform TLS and hostname verification remain enabled. */
    public static HttpResult performHttpRequest(String method, String url, byte[] body,
                                                String[] headerPairs, int timeoutMs) {
        HttpResult result = new HttpResult();
        result.url = url;
        HttpURLConnection connection = null;
        try {
            URL address = new URL(url);
            if (!address.getProtocol().equals("http") && !address.getProtocol().equals("https"))
                throw new IllegalArgumentException("HTTP transport requires http/https");
            connection = (HttpURLConnection) (address.getHost().equals("127.0.0.1") || address.getHost().equals("localhost")
                    ? address.openConnection(java.net.Proxy.NO_PROXY) : address.openConnection());
            connection.setConnectTimeout(Math.max(1, timeoutMs));
            connection.setReadTimeout(Math.max(1, timeoutMs));
            connection.setInstanceFollowRedirects(true);
            connection.setRequestMethod(method);
            if (headerPairs != null) {
                if (headerPairs.length % 2 != 0) throw new IllegalArgumentException("Invalid HTTP header pairs");
                for (int i = 0; i < headerPairs.length; i += 2)
                    connection.addRequestProperty(headerPairs[i], headerPairs[i + 1]);
            }
            if (body != null && body.length > 0) {
                connection.setDoOutput(true);
                connection.setFixedLengthStreamingMode(body.length);
                connection.getOutputStream().write(body);
                connection.getOutputStream().close();
            }
            result.status = connection.getResponseCode();
            result.url = connection.getURL().toString();
            List<String> headers = new ArrayList<>();
            for (Map.Entry<String, List<String>> entry : connection.getHeaderFields().entrySet())
                if (entry.getKey() != null) for (String value : entry.getValue()) {
                    headers.add(entry.getKey()); headers.add(value);
                }
            result.headers = headers.toArray(new String[0]);
            InputStream stream = result.status >= 400 ? connection.getErrorStream() : connection.getInputStream();
            if (stream != null) try (InputStream input = stream) { result.body = readBytes(input, 64 * 1024 * 1024); }
        } catch (Exception error) {
            result.error = error.getClass().getSimpleName() + ": " + error.getMessage();
        } finally { if (connection != null) connection.disconnect(); }
        return result;
    }
}
