package org.openpnp.vision.alternative;

import java.awt.image.BufferedImage;
import java.time.Duration;
import java.util.Locale;

import org.pmw.tinylog.Logger;

import org.openpnp.vision.alternative.BottomVisionEngine.Result;

/**
 * Shadow-mode observer for the local vision service.
 *
 * <p>Runs the same bottom-vision check the existing pipeline performs, but only
 * logs the result — it never affects alignment. The client is created lazily and
 * reused across calls. This class never throws: any failure (service down,
 * timeout, bad response) is logged and swallowed, so a shadow observation can
 * never break the machine.
 */
public final class ShadowBottomVisionObserver {
    private final int port;
    private final Duration timeout;
    private volatile LocalBottomVisionClient client;

    public ShadowBottomVisionObserver(int port, Duration timeout) {
        if (port < 1 || port > 65535) {
            throw new IllegalArgumentException("A valid port is required");
        }
        if (timeout == null || timeout.toMillis() < 1) {
            throw new IllegalArgumentException("A positive timeout is required");
        }
        this.port = port;
        this.timeout = timeout;
    }

    public int getPort() {
        return port;
    }

    public Duration getTimeout() {
        return timeout;
    }

    /**
     * Analyzes the image and logs the result. Never throws.
     */
    public void observe(BufferedImage image) {
        if (image == null) {
            return;
        }
        try {
            Result result = getClient().analyze(image);
            Logger.info("Shadow bottom vision: {}", format(result));
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Logger.warn("Shadow bottom vision: interrupted waiting for the service: {}", e.getMessage());
        }
        catch (Exception | Error e) {
            Logger.warn("Shadow bottom vision: observation failed: {}", e.getMessage());
        }
    }

    private LocalBottomVisionClient getClient() {
        LocalBottomVisionClient client = this.client;
        if (client == null) {
            synchronized (this) {
                client = this.client;
                if (client == null) {
                    client = new LocalBottomVisionClient(port, timeout);
                    this.client = client;
                }
            }
        }
        return client;
    }

    private static String format(Result result) {
        StringBuilder sb = new StringBuilder("status=").append(result.getStatus());
        result.getConfidence().ifPresent(confidence -> sb.append(String.format(Locale.ROOT,
                " confidence=%.2f", confidence)));
        result.getPose().ifPresent(pose -> sb.append(String.format(Locale.ROOT,
                " center=(%.1f,%.1f)px size=(%.1f,%.1f)px angle=%.1fdeg",
                pose.x, pose.y, pose.width, pose.height, pose.angle)));
        return sb.toString();
    }
}
