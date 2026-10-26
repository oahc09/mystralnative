package com.mystral.engine;

import android.os.Process;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.json.JSONArray;
import org.json.JSONObject;

/** Local APK assets and bounded OS-capture transport. Only the external Node collector judges pixels. */
final class InteropServer {
    private final MystralActivity activity;
    private final String runId;
    private final String caseId;
    private final String token = UUID.randomUUID().toString();
    private final File directory;
    private final ServerSocket socket = new ServerSocket();
    private final ExecutorService workers = Executors.newFixedThreadPool(4);
    private final Map<String, Capture> captures = new LinkedHashMap<>();
    private final Map<String, String> phases = new LinkedHashMap<>();
    private JSONObject runtime = new JSONObject();
    private volatile boolean closed;
    private volatile boolean reportSaved;
    private volatile boolean nativeEnded;
    private volatile int outcome = 1;
    private String origin;

    private static final class Capture {
        final String id = UUID.randomUUID().toString();
        final String kind;
        final String phase;
        final long deadline = SystemClock.elapsedRealtime() + 15000;
        final CountDownLatch acknowledged = new CountDownLatch(1);
        JSONObject metadata;
        JSONObject result;
        String error;
        boolean expired;
        Capture(String kind, String phase) { this.kind = kind; this.phase = phase; }
    }

    InteropServer(MystralActivity activity, String requestedRun, String requestedCase) throws Exception {
        this.activity = activity;
        runId = requestedRun == null ? "run-" + System.currentTimeMillis() + "-" + Process.myPid() : requestedRun;
        caseId = requestedCase == null ? "demo" : requestedCase;
        if (!runId.matches("[A-Za-z0-9_-]{1,100}")) throw new IllegalArgumentException("Invalid run_id");
        if (!Arrays.asList("demo", "probes", "air", "pbr", "skin", "ui", "compute").contains(caseId))
            throw new IllegalArgumentException("Invalid case_id");
        directory = new File(new File(activity.getFilesDir(), "interop"), runId);
        if (!directory.getParentFile().isDirectory() && !directory.getParentFile().mkdirs())
            throw new IllegalStateException("Cannot create private interop directory");
        if (!directory.mkdir()) throw new IllegalStateException("run_id must name a new attempt");
    }

    File directory() { return directory; }
    int outcome() { return outcome; }

    void start() throws Exception {
        socket.bind(new java.net.InetSocketAddress(InetAddress.getByName("127.0.0.1"), 0));
        origin = "http://127.0.0.1:" + socket.getLocalPort();
        writeNew(new File(directory, "session.json"), configuration().toString().getBytes(StandardCharsets.UTF_8));
        Thread listener = new Thread(() -> {
            while (!closed) {
                try {
                    Socket connection = socket.accept();
                    workers.execute(() -> serve(connection));
                } catch (Exception error) {
                    if (!closed) Log.e("MystralInterop", "Asset server accept failed", error);
                }
            }
        }, "InteropAssetServer");
        listener.setDaemon(true);
        listener.start();
        Log.i("MystralInterop", "SERVER_READY " + configuration().toString());
    }

    synchronized JSONObject configuration() throws Exception {
        return new JSONObject().put("schema", "air-android-interop-session-v1")
                .put("runId", runId).put("caseId", caseId).put("pid", Process.myPid())
                .put("demo", caseId.equals("demo"))
                .put("v8StartupMode", runtime.getString("v8StartupMode"))
                .put("v8Version", runtime.getString("v8Version"))
                .put("origin", origin).put("token", token)
                .put("entryUrl", origin + "/entry.js").put("bundleUrl", origin + "/sdk/cocosair.module.js")
                .put("assetBaseUrl", origin + "/sdk/webgpu/").put("assetRootUrl", origin + "/assets/")
                .put("moduleUrl", origin + "/probes/module-fixture.mjs")
                .put("captureUrl", origin + "/capture").put("reportUrl", origin + "/report")
                .put("eventUrl", origin + "/event");
    }

    synchronized void startup(String abi, String mode, String version) throws Exception {
        if ((!abi.equals("x86_64") && !abi.equals("arm64-v8a")) ||
                (!mode.equals("embedded") && !mode.equals("external")) || version.isEmpty())
            throw new IllegalArgumentException("Invalid actual native V8 startup identity");
        if (runtime.has("v8StartupMode") && (!abi.equals(runtime.getString("loadedABI")) ||
                !mode.equals(runtime.getString("v8StartupMode")) || !version.equals(runtime.getString("v8Version"))))
            throw new IllegalStateException("Native V8 startup identity changed");
        runtime.put("loadedABI", abi).put("v8StartupMode", mode).put("v8Version", version);
    }

    synchronized void snapshot(String abi, File file, String hash) throws Exception {
        if (!runtime.getString("v8StartupMode").equals("external"))
            throw new IllegalStateException("Embedded V8 must not request an external snapshot");
        runtime.put("loadedABI", abi).put("snapshotPath", file.getAbsolutePath()).put("snapshotSha256", hash);
        writeNew(new File(directory, "snapshot-receipt.json"), runtime.toString().getBytes(StandardCharsets.UTF_8));
    }

    synchronized void entry(File file, String hash) throws Exception {
        runtime.put("entryPath", file.getAbsolutePath()).put("entrySha256", hash);
        writeNew(new File(directory, "entry-receipt.json"), new JSONObject().put("file", file.getAbsolutePath())
                .put("sha256", hash).toString().getBytes(StandardCharsets.UTF_8));
    }

    synchronized void nativeReady(String abi, String engine, String backend, String mode, String version) {
        try {
            startup(abi, mode, version);
            runtime.put("loadedABI", abi).put("jsEngine", engine).put("backend", backend).put("nativeStarted", true);
            writeNew(new File(directory, "native-ready.json"), runtime.toString().getBytes(StandardCharsets.UTF_8));
            Log.i("MystralInterop", "NATIVE_READY " + runtime.toString());
        } catch (Exception error) { Log.e("MystralInterop", "Cannot record native identity", error); }
    }

    synchronized void nativeFinished(int code, String error) {
        nativeEnded = true;
        try {
            JSONObject result = new JSONObject().put("code", code).put("error", error)
                    .put("pid", Process.myPid()).put("elapsedRealtime", SystemClock.elapsedRealtime());
            writeNew(new File(directory, "native-exit.json"), result.toString().getBytes(StandardCharsets.UTF_8));
            Log.i("MystralInterop", "NATIVE_FINISHED " + result.toString());
        } catch (Exception failure) { Log.e("MystralInterop", "Cannot record native exit", failure); }
    }

    synchronized void lifecycle(String event, int generation, int orientation) {
        try {
            JSONObject row = new JSONObject().put("event", event).put("pid", Process.myPid())
                    .put("surfaceGeneration", generation).put("orientation", orientation)
                    .put("elapsedRealtime", SystemClock.elapsedRealtime());
            try (FileOutputStream output = new FileOutputStream(new File(directory, "lifecycle.jsonl"), true)) {
                output.write((row.toString() + "\n").getBytes(StandardCharsets.UTF_8));
            }
            Log.i("MystralInterop", "LIFECYCLE " + row.toString());
        } catch (Exception error) { Log.e("MystralInterop", "Cannot retain lifecycle event", error); }
    }

    private synchronized JSONObject state() throws Exception {
        JSONArray values = new JSONArray();
        for (Capture capture : captures.values()) {
            JSONObject row = new JSONObject().put("captureId", capture.id).put("kind", capture.kind)
                    .put("phase", capture.phase).put("deadlineElapsedRealtime", capture.deadline)
                    .put("expired", capture.expired);
            if (capture.metadata != null) row.put("metadata", capture.metadata);
            if (capture.result != null) row.put("acknowledged", true);
            if (capture.error != null) row.put("error", capture.error);
            values.put(row);
        }
        return new JSONObject().put("session", configuration()).put("runtime", new JSONObject(runtime.toString()))
                .put("captures", values).put("reportSaved", reportSaved).put("nativeFinished", nativeEnded);
    }

    private synchronized Capture reserve(JSONObject request) throws Exception {
        String kind = request.getString("kind");
        String expected = caseId.equals("probes") || caseId.equals("air") ? caseId : "scene-" + caseId;
        if (!kind.equals(expected)) throw new IllegalArgumentException("Capture kind does not belong to this process");
        String phase = request.optString("phase", "");
        boolean scene = kind.startsWith("scene-");
        if (scene ? !phase.equals("before") && !phase.equals("after") : !phase.isEmpty())
            throw new IllegalArgumentException("Invalid capture phase");
        String key = kind + ":" + phase;
        if (phases.containsKey(key)) throw new IllegalStateException("Capture phase was already attempted");
        if (phase.equals("after")) {
            Capture before = captures.get(phases.get(kind + ":before"));
            if (before == null || before.result == null || before.expired || before.error != null)
                throw new IllegalStateException("Before capture is missing its independent acknowledgement");
        }
        Capture capture = new Capture(kind, phase);
        captures.put(capture.id, capture); phases.put(key, capture.id);
        return capture;
    }

    private JSONObject capture(byte[] body) throws Exception {
        Capture capture = reserve(new JSONObject(new String(body, StandardCharsets.UTF_8)));
        String name = capture.kind + (capture.phase.isEmpty() ? "" : "-" + capture.phase);
        writeNew(new File(directory, name + ".request.json"), body);
        try {
            JSONObject metadata = activity.captureSurface(new File(directory, name + ".png"), capture.deadline);
            metadata.put("captureId", capture.id).put("kind", capture.kind).put("phase", capture.phase)
                    .put("runId", runId);
            synchronized (this) {
                if (capture.phase.equals("after")) {
                    Capture before = captures.get(phases.get(capture.kind + ":before"));
                    if (before.metadata.getInt("surfaceGeneration") != metadata.getInt("surfaceGeneration") ||
                            before.metadata.getInt("pid") != metadata.getInt("pid"))
                        throw new IllegalStateException("Surface identity changed between scene phases");
                    metadata.put("previous", new JSONObject().put("file", before.metadata.getString("file"))
                            .put("sha256", before.metadata.getString("sha256"))
                            .put("pid", before.metadata.getInt("pid"))
                            .put("surfaceGeneration", before.metadata.getInt("surfaceGeneration")));
                }
            }
            writeNew(new File(directory, name + ".capture.json"), metadata.toString().getBytes(StandardCharsets.UTF_8));
            synchronized (this) { capture.metadata = metadata; }
            long remaining = capture.deadline - SystemClock.elapsedRealtime();
            if (remaining <= 0 || !capture.acknowledged.await(remaining, TimeUnit.MILLISECONDS))
                throw new IllegalStateException("Original fifteen-second capture/ack budget exceeded");
            synchronized (this) {
                if (capture.result == null || capture.expired) throw new IllegalStateException("Missing independent capture verdict");
                JSONObject reply = new JSONObject(capture.result.toString());
                for (java.util.Iterator<String> keys = metadata.keys(); keys.hasNext();) {
                    String key = keys.next(); reply.put(key, metadata.get(key));
                }
                return reply;
            }
        } catch (Exception error) {
            synchronized (this) { capture.expired = true; capture.error = error.toString(); }
            writeNew(new File(directory, name + ".failure.json"),
                    new JSONObject().put("error", error.toString()).put("captureId", capture.id)
                            .toString().getBytes(StandardCharsets.UTF_8));
            throw error;
        }
    }

    private synchronized JSONObject acknowledge(byte[] body) throws Exception {
        JSONObject request = new JSONObject(new String(body, StandardCharsets.UTF_8));
        Capture capture = captures.get(request.getString("captureId"));
        if (capture == null || capture.metadata == null || capture.expired || capture.result != null ||
                SystemClock.elapsedRealtime() >= capture.deadline)
            throw new IllegalStateException("Capture acknowledgement is stale or unrequested");
        if (!capture.metadata.getString("sha256").equals(request.getString("sha256")))
            throw new IllegalArgumentException("Acknowledgement does not bind the captured PNG");
        JSONObject verdict = request.getJSONObject("result");
        if (!(verdict.get("pass") instanceof Boolean) || !Arrays.asList("PASS", "FAIL").contains(verdict.getString("status")))
            throw new IllegalArgumentException("Independent verdict must contain pass/status");
        if (verdict.getBoolean("pass") != verdict.getString("status").equals("PASS"))
            throw new IllegalArgumentException("Independent verdict pass/status contradict each other");
        if (verdict.has("sha256") && !verdict.getString("sha256").equals(request.getString("sha256")))
            throw new IllegalArgumentException("Oracle and capture image hashes differ");
        writeNew(new File(directory, capture.kind + (capture.phase.isEmpty() ? "" : "-" + capture.phase) + ".ack.json"), body);
        capture.result = verdict; capture.acknowledged.countDown();
        return new JSONObject().put("acknowledged", capture.id);
    }

    private void serve(Socket connection) {
        try {
            Socket client = connection;
            client.setSoTimeout(20000);
            InputStream input = new BufferedInputStream(client.getInputStream());
            String[] request = line(input).split(" ");
            if (request.length != 3) throw new IllegalArgumentException("Malformed HTTP request");
            Map<String, String> headers = new LinkedHashMap<>();
            String header;
            int headerBytes = 0;
            while (!(header = line(input)).isEmpty()) {
                headerBytes += header.length();
                if (headerBytes > 65536) throw new IllegalArgumentException("HTTP headers exceed bound");
                int colon = header.indexOf(':');
                if (colon <= 0) throw new IllegalArgumentException("Malformed HTTP header");
                headers.put(header.substring(0, colon).toLowerCase(Locale.ROOT), header.substring(colon + 1).trim());
            }
            if (headers.containsKey("transfer-encoding")) throw new IllegalArgumentException("Use explicit Content-Length");
            int length = Integer.parseInt(headers.containsKey("content-length") ? headers.get("content-length") : "0");
            if (length < 0 || length > 16 * 1024 * 1024) throw new IllegalArgumentException("HTTP body exceeds bound");
            byte[] body = new byte[length];
            for (int offset = 0; offset < length;) {
                int count = input.read(body, offset, length - offset);
                if (count < 0) throw new IllegalArgumentException("Incomplete HTTP body");
                offset += count;
            }
            String route = URLDecoder.decode(request[1].split("\\?", 2)[0], "UTF-8");
            boolean control = Arrays.asList("/state", "/capture", "/ack", "/report", "/event", "/finish").contains(route) || route.startsWith("/artifact/");
            if (control && !token.equals(headers.get("x-interop-token"))) {
                send(client, 403, "application/json", "{\"error\":\"Invalid interop token\"}".getBytes(StandardCharsets.UTF_8), false);
                return;
            }
            if (request[0].equals("GET") && route.equals("/state")) json(client, state());
            else if (request[0].equals("POST") && route.equals("/capture")) json(client, capture(body));
            else if (request[0].equals("POST") && route.equals("/ack")) json(client, acknowledge(body));
            else if (request[0].equals("POST") && route.equals("/report")) {
                JSONObject report = new JSONObject(new String(body, StandardCharsets.UTF_8));
                writeNew(new File(directory, "report.json"), body);
                outcome = report.has("pass") ? (report.optBoolean("pass", false) ? 0 : 1)
                        : (report.optString("status", "FAIL").equals("PASS") ? 0 : 1);
                if (report.has("status") && !report.optString("status").equals("PASS")) outcome = 1;
                reportSaved = true;
                Log.i("MystralInterop", "REPORT_SAVED " + runId);
                json(client, new JSONObject().put("saved", true));
            } else if (request[0].equals("POST") && route.equals("/event")) {
                JSONObject event = new JSONObject(new String(body, StandardCharsets.UTF_8));
                synchronized (this) {
                    try (FileOutputStream output = new FileOutputStream(new File(directory, "events.jsonl"), true)) {
                        output.write(body); output.write('\n');
                    }
                }
                Log.i("MystralInterop", "EVENT " + event.optString("stage", "event"));
                json(client, new JSONObject().put("saved", true));
            } else if (request[0].equals("POST") && route.equals("/finish")) {
                if (!reportSaved) throw new IllegalStateException("A report must be saved before finishing");
                Log.i("MystralInterop", "EXIT_STAGE finish-request run=" + runId + " outcome=" + outcome);
                json(client, new JSONObject().put("finishing", true));
                Log.i("MystralInterop", "EXIT_STAGE finish-ack-written run=" + runId);
                activity.requestNativeQuit();
                Log.i("MystralInterop", "EXIT_STAGE finish-quit-scheduled run=" + runId);
            } else if (request[0].equals("GET") && route.startsWith("/artifact/")) {
                String name = route.substring("/artifact/".length());
                if (!name.matches("[A-Za-z0-9_.-]+")) throw new IllegalArgumentException("Invalid artifact name");
                File file = new File(directory, name);
                if (!file.isFile()) { send(client, 404, "text/plain", new byte[0], false); return; }
                try (InputStream stream = new java.io.FileInputStream(file)) {
                    send(client, 200, mime(name), MystralActivity.readBytes(stream, 64 * 1024 * 1024), false);
                }
            } else if (request[0].equals("GET") || request[0].equals("HEAD")) {
                if (route.contains("..") || route.contains("\\") || route.indexOf('\0') >= 0)
                    throw new IllegalArgumentException("Invalid asset path");
                try (InputStream stream = activity.getAssets().open("interop" + route)) {
                    send(client, 200, mime(route), MystralActivity.readBytes(stream, 64 * 1024 * 1024), request[0].equals("HEAD"));
                } catch (java.io.FileNotFoundException error) {
                    send(client, 404, "text/plain", error.toString().getBytes(StandardCharsets.UTF_8), false);
                }
            } else throw new IllegalArgumentException("Unsupported endpoint/method");
        } catch (Exception error) {
            try { send(connection, 500, "application/json", new JSONObject().put("error", error.toString())
                    .put("pass", false).put("status", "FAIL").toString().getBytes(StandardCharsets.UTF_8), false); }
            catch (Exception ignored) { }
            Log.e("MystralInterop", "Request failed: " + error.toString());
        } finally {
            try { connection.close(); } catch (Exception ignored) { }
        }
    }

    private static String line(InputStream input) throws Exception {
        StringBuilder value = new StringBuilder();
        int next;
        while ((next = input.read()) != -1 && next != '\n') {
            if (next != '\r') value.append((char) next);
            if (value.length() > 8192) throw new IllegalArgumentException("HTTP line exceeds bound");
        }
        if (next == -1) throw new IllegalArgumentException("HTTP connection ended before line");
        return value.toString();
    }

    private static String mime(String path) {
        if (path.endsWith(".wasm")) return "application/wasm";
        if (path.endsWith(".js") || path.endsWith(".mjs")) return "text/javascript; charset=utf-8";
        if (path.endsWith(".json") || path.endsWith(".gltf")) return "application/json";
        if (path.endsWith(".png")) return "image/png";
        if (path.endsWith(".glb")) return "model/gltf-binary";
        return "application/octet-stream";
    }

    private static void json(Socket socket, JSONObject value) throws Exception {
        send(socket, 200, "application/json", value.toString().getBytes(StandardCharsets.UTF_8), false);
    }

    private static void send(Socket socket, int status, String type, byte[] bytes, boolean head) throws Exception {
        OutputStream output = socket.getOutputStream();
        output.write(("HTTP/1.1 " + status + " Result\r\nContent-Type: " + type + "\r\nContent-Length: " + bytes.length +
                "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
        if (!head) output.write(bytes);
        output.flush();
    }

    static void writeNew(File file, byte[] bytes) throws Exception {
        if (!file.createNewFile()) throw new IllegalStateException("Refusing to replace artifact: " + file.getName());
        try (FileOutputStream output = new FileOutputStream(file)) { output.write(bytes); }
    }

    static String sha256(byte[] bytes) throws Exception {
        StringBuilder value = new StringBuilder();
        for (byte part : MessageDigest.getInstance("SHA-256").digest(bytes)) value.append(String.format(Locale.ROOT, "%02x", part & 255));
        return value.toString();
    }

    void close() {
        closed = true;
        try { socket.close(); } catch (Exception ignored) { }
        synchronized (this) {
            for (Capture capture : captures.values()) { capture.expired = true; capture.acknowledged.countDown(); }
        }
        workers.shutdownNow();
    }
}
