package org.openpnp.vision.alternative;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.imageio.ImageIO;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;

/**
 * Calls a separate process on 127.0.0.1. POST /v1/bottom-vision accepts a PNG and returns
 * JSON with status and optional confidence. A detected result also has center_px,
 * size_px (two numbers each), and angle_deg. Other results carry no usable pose.
 */
public final class LocalBottomVisionClient implements BottomVisionEngine {
    private final HttpClient client;
    private final URI endpoint;
    private final Duration timeout;

    public LocalBottomVisionClient(int port, Duration timeout) {
        this.timeout = Objects.requireNonNull(timeout, "timeout");
        if (port < 1 || port > 65535 || timeout.toMillis() < 1) {
            throw new IllegalArgumentException("A valid port and positive timeout are required");
        }
        this.endpoint = URI.create("http://127.0.0.1:" + port + "/v1/bottom-vision");
        this.client = HttpClient.newBuilder().connectTimeout(timeout)
                .proxy(ProxySelector.of(null)).build();
    }

    @Override
    public Result analyze(BufferedImage image) throws IOException, InterruptedException {
        Objects.requireNonNull(image, "image");
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        if (!ImageIO.write(image, "PNG", png)) {
            throw new IOException("No PNG image writer is available");
        }
        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .header("Content-Type", "image/png")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofByteArray(png.toByteArray()))
                .build();
        CompletableFuture<HttpResponse<String>> pending =
                client.sendAsync(request, HttpResponse.BodyHandlers.ofString());
        try {
            HttpResponse<String> response = pending.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
            if (response.statusCode() != 200) {
                throw new IOException("Bottom vision service returned HTTP " + response.statusCode());
            }
            return decode(response.body(), image.getWidth(), image.getHeight());
        }
        catch (ExecutionException e) {
            throw new IOException("Bottom vision service request failed", e.getCause());
        }
        catch (TimeoutException e) {
            throw new IOException("Bottom vision service timed out after " + timeout.toMillis() + " ms", e);
        }
        finally {
            if (!pending.isDone()) {
                pending.cancel(true);
            }
        }
    }

    private static Result decode(String body, int width, int height) throws IOException {
        try {
            JsonObject json = new JsonParser().parse(body).getAsJsonObject();
            JsonElement statusValue = json.get("status");
            if (statusValue == null || !statusValue.isJsonPrimitive()
                    || !statusValue.getAsJsonPrimitive().isString()) {
                throw new IllegalArgumentException("Expected a status string");
            }
            Status status = Status.valueOf(statusValue.getAsString().toUpperCase(Locale.ROOT));
            JsonElement confidenceValue = json.get("confidence");
            Double confidence = confidenceValue == null || confidenceValue.isJsonNull()
                    ? null : number(confidenceValue);
            Pose pose = null;
            if (status == Status.DETECTED) {
                JsonArray center = json.getAsJsonArray("center_px");
                JsonArray size = json.getAsJsonArray("size_px");
                if (center == null || center.size() != 2 || size == null || size.size() != 2) {
                    throw new IllegalArgumentException("Detected result requires center_px and size_px pairs");
                }
                pose = new Pose(number(center.get(0)), number(center.get(1)),
                        number(size.get(0)), number(size.get(1)), number(json.get("angle_deg")));
                if (pose.x >= width || pose.y >= height) {
                    throw new IllegalArgumentException("Detected center is outside the supplied image");
                }
            }
            return new Result(status, pose, confidence);
        }
        catch (JsonParseException | IllegalArgumentException | IllegalStateException
                | UnsupportedOperationException | ClassCastException e) {
            throw new IOException("Invalid bottom vision service response", e);
        }
    }

    private static double number(JsonElement value) {
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("Expected a JSON number");
        }
        return value.getAsDouble();
    }
}
