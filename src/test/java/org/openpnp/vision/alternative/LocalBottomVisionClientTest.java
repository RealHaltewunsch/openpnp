package org.openpnp.vision.alternative;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openpnp.vision.alternative.BottomVisionEngine.Pose;
import org.openpnp.vision.alternative.BottomVisionEngine.Result;
import org.openpnp.vision.alternative.BottomVisionEngine.Status;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpServer;

class LocalBottomVisionClientTest {
    private HttpServer server;
    private BufferedImage image;

    @BeforeEach
    void createServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        image = new BufferedImage(40, 24, BufferedImage.TYPE_INT_RGB);
        image.setRGB(10, 12, 0x315806);
    }

    @AfterEach
    void stopServer() {
        server.stop(0);
    }

    @Test
    void transportsImageAndDecodesOriginalPixelCoordinates() throws Exception {
        Result result = analyze("{\"status\":\"detected\",\"center_px\":[10.25,12.5],"
                + "\"size_px\":[8,4],\"angle_deg\":-42.5,\"confidence\":0.91}");
        assertEquals(Status.DETECTED, result.getStatus());
        Pose pose = result.getPose().orElseThrow();
        assertEquals(10.25, pose.x);
        assertEquals(12.5, pose.y);
        assertEquals(8, pose.width);
        assertEquals(4, pose.height);
        assertEquals(-42.5, pose.angle);
        assertEquals(0.91, result.getConfidence().orElseThrow());
    }

    @Test
    void uncertainResultCannotExposeAPose() throws Exception {
        Result result = analyze("{\"status\":\"uncertain\",\"confidence\":0.25,\"center_px\":[10,12]}");
        assertEquals(Status.UNCERTAIN, result.getStatus());
        assertTrue(result.getPose().isEmpty());
    }

    @Test
    void rejectsAReportedCenterOutsideTheInputFrame() {
        assertThrows(IOException.class, () -> analyze("{\"status\":\"detected\","
                + "\"center_px\":[40,12],\"size_px\":[8,4],\"angle_deg\":0}"));
    }

    @Test
    void timesOutWhenResponseBodyDoesNotFinish() {
        CountDownLatch releaseServer = new CountDownLatch(1);
        CountDownLatch headersSent = new CountDownLatch(1);
        server.createContext("/v1/infer", exchange -> {
            try {
                exchange.getRequestBody().readAllBytes();
                exchange.sendResponseHeaders(200, 0);
                exchange.getResponseBody().write('{');
                exchange.getResponseBody().flush();
                headersSent.countDown();
                releaseServer.await(10, TimeUnit.SECONDS);
            }
            catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            finally {
                exchange.close();
            }
        });
        server.start();
        try {
            LocalBottomVisionClient client =
                    new LocalBottomVisionClient(server.getAddress().getPort(), Duration.ofSeconds(2));
            IOException error = assertThrows(IOException.class, () -> client.analyze(image));
            assertEquals(0, headersSent.getCount());
            assertTrue(error.getMessage().contains("timed out"));
        }
        finally {
            releaseServer.countDown();
        }
    }

    private Result analyze(String json) throws IOException, InterruptedException {
        server.createContext("/v1/infer", exchange -> {
            try {
                JsonObject request = new JsonParser().parse(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject();
                String imageB64 = request.get("image_png_base64").getAsString();
                BufferedImage received = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder().decode(imageB64)));
                if (!"POST".equals(exchange.getRequestMethod())
                        || !"application/json".equals(exchange.getRequestHeaders().getFirst("Content-Type"))
                        || !"bottom-vision".equals(request.get("task").getAsString())
                        || received == null || received.getWidth() != image.getWidth()
                        || received.getHeight() != image.getHeight()
                        || received.getRGB(10, 12) != image.getRGB(10, 12)) {
                    exchange.sendResponseHeaders(400, -1);
                    return;
                }
                byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            }
            finally {
                exchange.close();
            }
        });
        server.start();
        return new LocalBottomVisionClient(server.getAddress().getPort(), Duration.ofSeconds(5)).analyze(image);
    }
}
