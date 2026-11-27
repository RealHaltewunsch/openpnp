package org.openpnp.vision.service;

import static org.junit.jupiter.api.Assertions.*;

import java.net.InetAddress;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import java.util.prefs.Preferences;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

class VisionServiceControllerTest {
    private Preferences preferences;
    private VisionServiceController controller;

    @BeforeEach
    void isolatedPreferences() {
        preferences = Preferences.userRoot().node("org.openpnp.tests.vision." + UUID.randomUUID());
        controller = new VisionServiceController(preferences);
    }

    @AfterEach
    void cleanup() throws Exception {
        controller.stop();
        preferences.removeNode();
    }

    @Test
    void preferencesPreservePathsPortAndExplicitEnablement() throws Exception {
        Path service = Files.createTempDirectory("vision-service-preferences-");
        assertFalse(controller.isEnabled());
        controller.configure(service, service.resolve("custom weights"), 8021, "custom-python");
        controller.setEnabled(true);
        VisionServiceController restored = new VisionServiceController(preferences);
        assertEquals(service, restored.getServiceDir());
        assertEquals(service.resolve("custom weights"), restored.getModelsDir());
        assertEquals(8021, restored.getPort());
        assertEquals("custom-python", restored.getPython());
        assertTrue(restored.isEnabled());
        assertFalse(restored.isRunning());
        assertThrows(IllegalArgumentException.class,
                () -> controller.configure(service, service, 0, "python3"));
    }

    @Test
    void readinessRequiresCorrectContractAndAvailableBottomVision() {
        String health = "{\"service\":\"openpnp-vision\",\"contract_version\":1,"
                + "\"encoder_version\":\"test\",\"heads\":{\"bottom-vision\":{\"status\":\"ok\"},"
                + "\"homing\":{\"status\":\"missing_weights\"}}}";
        assertTrue(VisionServiceController.isBottomVisionReady(health));
        assertFalse(VisionServiceController.isBottomVisionReady(health.replace("\"ok\"", "\"load_error\"")));
        assertFalse(VisionServiceController.isBottomVisionReady(health.replace("\"encoder_version\":\"test\"", "\"encoder_version\":null")));
        assertFalse(VisionServiceController.isBottomVisionReady(health.replace("\"contract_version\":1", "\"contract_version\":2")));
    }

    @Test
    void ownsStartupAndShutdownAndRejectsReconfigurationWhileRunning() throws Exception {
        String python = controller.getPython();
        try {
            Assumptions.assumeTrue(new ProcessBuilder(python, "--version").start().waitFor() == 0);
        }
        catch (java.io.IOException e) {
            Assumptions.assumeTrue(false, "Python is optional for Java-only test environments");
        }
        Path service = Files.createTempDirectory("vision service with spaces-");
        Files.createDirectories(service.resolve("vision_service"));
        Files.createDirectories(service.resolve("model weights"));
        Files.writeString(service.resolve("model weights/manifest.json"), "{}");
        Files.writeString(service.resolve("vision_service/__main__.py"),
                "import argparse, json\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n"
                + "p=argparse.ArgumentParser()\np.add_argument('--models-dir')\np.add_argument('--host')\np.add_argument('--port',type=int)\na=p.parse_args()\n"
                + "class Handler(BaseHTTPRequestHandler):\n def do_GET(self):\n"
                + "  data=json.dumps({'service':'openpnp-vision','contract_version':1,'encoder_version':'test',"
                + "'heads':{'bottom-vision':{'status':'ok'}}}).encode()\n"
                + "  self.send_response(200)\n  self.end_headers()\n  self.wfile.write(data)\n"
                + "HTTPServer((a.host,a.port),Handler).serve_forever()\n");
        int port;
        try (ServerSocket free = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
            port = free.getLocalPort();
        }
        controller.configure(service, service.resolve("model weights"), port, python);
        try (ServerSocket occupied = new ServerSocket(port, 1, InetAddress.getByName("127.0.0.1"))) {
            assertThrows(java.net.BindException.class, controller::start);
            assertFalse(controller.isRunning());
        }
        controller.start();
        assertTrue(controller.isRunning());
        controller.start(); // idempotent
        assertThrows(IllegalStateException.class,
                () -> controller.configure(service, service, port, python));
        controller.stop();
        assertFalse(controller.isRunning());
        try (ServerSocket reusable = new ServerSocket(port, 1, InetAddress.getByName("127.0.0.1"))) {
            assertEquals(port, reusable.getLocalPort());
        }
        Files.writeString(service.resolve("vision_service/__main__.py"), "raise SystemExit(7)\n");
        assertThrows(java.io.IOException.class, controller::start);
        assertFalse(controller.isRunning());
    }
}
