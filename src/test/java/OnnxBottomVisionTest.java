import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Duration;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.Test;
import org.openpnp.machine.reference.ReferenceNozzleTip;
import org.openpnp.machine.reference.vision.ReferenceBottomVision;
import org.openpnp.model.Configuration;
import org.openpnp.model.Length;
import org.openpnp.model.LengthUnit;
import org.openpnp.model.Location;
import org.openpnp.model.OnnxBottomVisionSettings;
import org.openpnp.model.Part;
import org.openpnp.model.Placement;
import org.openpnp.spi.Machine;
import org.openpnp.spi.Nozzle;
import org.openpnp.spi.PartAlignment.PartAlignmentOffset;
import org.openpnp.vision.alternative.BottomVisionEngine;
import org.openpnp.vision.alternative.LocalBottomVisionClient;
import org.openpnp.vision.pipeline.CvPipeline;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpServer;

class OnnxBottomVisionTest {
    @Test
    void servicePoseDrivesPostRotationWithoutAnOpenCvPipeline() throws Exception {
        testAlignment(false, false);
    }

    @Test
    void servicePoseDrivesIterativePreRotationWithoutAnOpenCvPipeline() throws Exception {
        testAlignment(true, false);
    }

    @Test
    void uncertainServiceResultStopsAlignment() throws Exception {
        testAlignment(false, true);
    }

    private void testAlignment(boolean preRotate, boolean uncertain) throws Exception {
        Configuration.initialize(Files.createTempDirectory("onnx-alignment-").toFile());
        Configuration.get().load();
        SampleJobTest.makeMachineFastest();
        Machine machine = Configuration.get().getMachine();
        Nozzle nozzle = machine.getDefaultHead().getDefaultNozzle();
        Part part = Configuration.get().getPart("R0805-1K");
        ((ReferenceNozzleTip) nozzle.getNozzleTip()).setMaxPickTolerance(new Length(5, LengthUnit.Millimeters));
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        AtomicInteger calls = new AtomicInteger();
        AtomicReference<Throwable> serverError = new AtomicReference<>();
        server.createContext("/v1/infer", exchange -> {
            try {
                JsonObject request = new JsonParser().parse(new String(exchange.getRequestBody().readAllBytes(),
                        StandardCharsets.UTF_8)).getAsJsonObject();
                assertEquals("bottom-vision", request.get("task").getAsString());
                BufferedImage image = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder()
                        .decode(request.get("image_png_base64").getAsString())));
                assertNotNull(image);
                int pass = calls.incrementAndGet();
                // Force another pre-rotation pass, then report a centered part.
                double dx = !preRotate || pass == 1 ? 2 : 0;
                String body = uncertain ? "{\"status\":\"uncertain\"}" : String.format(java.util.Locale.ROOT,
                        "{\"status\":\"detected\",\"center_px\":[%.1f,%.1f],\"size_px\":[20,10],\"angle_deg\":0}",
                        image.getWidth() / 2.0 + dx, image.getHeight() / 2.0);
                byte[] response = body.getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, response.length);
                exchange.getResponseBody().write(response);
            }
            catch (Throwable error) {
                serverError.set(error);
            }
            finally {
                exchange.close();
            }
        });
        server.start();
        ReferenceBottomVision alignment = new ReferenceBottomVision() {
            @Override
            protected BottomVisionEngine createOnnxEngine() {
                return new LocalBottomVisionClient(server.getAddress().getPort(), Duration.ofSeconds(2));
            }
        };
        alignment.setEnabled(true);
        alignment.setPreRotate(preRotate);
        alignment.setMaxLinearOffset(new Length(0.001, LengthUnit.Millimeters));
        OnnxBottomVisionSettings settings = new OnnxBottomVisionSettings() {
            @Override
            public CvPipeline getPipeline() {
                throw new AssertionError("ONNX alignment must not access an OpenCV pipeline");
            }
        };
        settings.resetToDefault();
        part.setBottomVisionSettings(settings);
        Placement placement = new Placement("test");
        placement.setLocation(Location.origin);
        try {
            machine.setEnabled(true);
            machine.home();
            machine.execute(() -> {
                nozzle.pick(part, null);
                try {
                    if (uncertain) {
                        Exception error = assertThrows(Exception.class,
                                () -> alignment.findOffsets(part, null, placement, nozzle));
                        assertTrue(error.getMessage().contains("UNCERTAIN"));
                    }
                    else {
                        PartAlignmentOffset offset = alignment.findOffsets(part, null, placement, nozzle);
                        assertEquals(preRotate, offset.getPreRotated());
                        assertNotEquals(0, offset.getLocation().getX(), 0.00001);
                        assertEquals(0, offset.getLocation().getY(), 0.00001);
                        assertEquals(0, offset.getLocation().getRotation(), 0.00001);
                    }
                }
                finally {
                    nozzle.place();
                }
                return true;
            });
            assertNull(serverError.get());
            assertEquals(preRotate ? 2 : 1, calls.get());
        }
        finally {
            server.stop(0);
            machine.setEnabled(false);
            machine.close();
        }
    }
}
