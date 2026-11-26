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

import javax.imageio.ImageIO;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openpnp.machine.reference.vision.ReferenceBottomVision;
import org.openpnp.model.Configuration;
import org.openpnp.util.XmlSerialize;
import org.simpleframework.xml.Element;
import org.simpleframework.xml.Serializer;

import com.google.common.io.Files;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpServer;

class ShadowBottomVisionObserverTest {
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
    void neverThrowsWhenServiceIsDown() {
        server.stop(0);
        int port = server.getAddress().getPort();
        ShadowBottomVisionObserver observer = new ShadowBottomVisionObserver(port, Duration.ofSeconds(2));
        observer.observe(image); // must not throw
    }

    @Test
    void neverThrowsWhenServiceReturnsAnError() throws IOException {
        server.createContext("/v1/infer", exchange -> {
            try {
                exchange.sendResponseHeaders(500, -1);
            } finally {
                exchange.close();
            }
        });
        server.start();
        ShadowBottomVisionObserver observer =
                new ShadowBottomVisionObserver(server.getAddress().getPort(), Duration.ofSeconds(5));
        observer.observe(image); // must not throw
    }

    @Test
    void neverThrowsWhenServiceReturnsGarbage() throws IOException {
        server.createContext("/v1/infer", exchange -> {
            try {
                byte[] bytes = "not json at all".getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            } finally {
                exchange.close();
            }
        });
        server.start();
        ShadowBottomVisionObserver observer =
                new ShadowBottomVisionObserver(server.getAddress().getPort(), Duration.ofSeconds(5));
        observer.observe(image); // must not throw
    }

    @Test
    void observesAgainstAMockServerWithoutThrowing() throws IOException {
        server.createContext("/v1/infer", exchange -> {
            try {
                JsonObject request = new JsonParser().parse(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject();
                String imageB64 = request.get("image_png_base64").getAsString();
                BufferedImage received = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder().decode(imageB64)));
                if (received == null || received.getWidth() != image.getWidth()) {
                    exchange.sendResponseHeaders(400, -1);
                    return;
                }
                byte[] bytes = ("{\"status\":\"detected\",\"center_px\":[10,12],"
                        + "\"size_px\":[8,4],\"angle_deg\":-42.5,\"confidence\":0.91}")
                        .getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            } finally {
                exchange.close();
            }
        });
        server.start();
        ShadowBottomVisionObserver observer =
                new ShadowBottomVisionObserver(server.getAddress().getPort(), Duration.ofSeconds(5));
        observer.observe(image); // must not throw
    }

    @Test
    void ignoresANullImage() {
        ShadowBottomVisionObserver observer = new ShadowBottomVisionObserver(1, Duration.ofSeconds(1));
        observer.observe(null); // must not throw
    }

    @Test
    void validatesPortAndTimeout() {
        assertThrows(IllegalArgumentException.class,
                () -> new ShadowBottomVisionObserver(0, Duration.ofSeconds(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ShadowBottomVisionObserver(70000, Duration.ofSeconds(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ShadowBottomVisionObserver(1, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ShadowBottomVisionObserver(1, Duration.ZERO));
    }

    @Test
    void xmlRoundTripPreservesShadowAttributes() throws Exception {
        Configuration.initialize(Files.createTempDir());
        ReferenceBottomVision alignment = new ReferenceBottomVision();
        alignment.setShadowMode(true);
        alignment.setShadowPort(8021);
        alignment.setShadowTimeoutMs(250);

        Serializer serializer = XmlSerialize.createSerializer();
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        serializer.write(new Holder(alignment), out);
        String xml = out.toString(java.nio.charset.StandardCharsets.UTF_8);
        assertTrue(xml.contains("shadow-mode=\"true\""), "xml: " + xml);
        assertTrue(xml.contains("shadow-port=\"8021\""), "xml: " + xml);
        assertTrue(xml.contains("shadow-timeout-ms=\"250\""), "xml: " + xml);

        Holder roundTripped = serializer.read(Holder.class, new java.io.ByteArrayInputStream(xml.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        ReferenceBottomVision restored = roundTripped.alignment;
        assertTrue(restored.isShadowMode());
        assertEquals(8021, restored.getShadowPort());
        assertEquals(250, restored.getShadowTimeoutMs());
    }

    @Test
    void shadowModeDefaultsToOff() throws Exception {
        Configuration.initialize(Files.createTempDir());
        ReferenceBottomVision alignment = new ReferenceBottomVision();
        assertEquals(false, alignment.isShadowMode());
        assertEquals(8010, alignment.getShadowPort());
        assertEquals(1000, alignment.getShadowTimeoutMs());
    }

    static class Holder {
        @Element(required = false)
        ReferenceBottomVision alignment;

        Holder() {
        }

        Holder(ReferenceBottomVision alignment) {
            this.alignment = alignment;
        }
    }
}
