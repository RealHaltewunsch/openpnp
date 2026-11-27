package org.openpnp.vision.service;

import java.awt.Component;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ProxySelector;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.prefs.Preferences;

import javax.swing.JCheckBoxMenuItem;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.SwingWorker;
import javax.swing.Timer;

import org.openpnp.Translations;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/** Owns the local service process; never launches a shell or exposes a non-loopback listener. */
public final class VisionServiceController {
    private static final VisionServiceController INSTANCE = new VisionServiceController();
    private final Preferences preferences;
    private volatile Process process;
    private volatile boolean ready;

    private VisionServiceController() {
        this(Preferences.userRoot().node("org.openpnp.vision.service"));
        Runtime.getRuntime().addShutdownHook(new Thread(this::stop, "vision-service-shutdown"));
    }

    VisionServiceController(Preferences preferences) {
        this.preferences = preferences;
    }

    public static VisionServiceController getInstance() {
        return INSTANCE;
    }

    public boolean isEnabled() {
        return preferences.getBoolean("enabled", false);
    }

    public void setEnabled(boolean enabled) {
        preferences.putBoolean("enabled", enabled);
    }

    public boolean isRunning() {
        Process current = process;
        return ready && current != null && current.isAlive();
    }

    public Path getServiceDir() {
        return Path.of(preferences.get("serviceDir", Path.of(System.getProperty("user.dir"), "vision-service").toString()));
    }

    public Path getModelsDir() {
        return Path.of(preferences.get("modelsDir", getServiceDir().resolve("models/synthetic").toString()));
    }

    public int getPort() {
        return preferences.getInt("port", 8010);
    }

    public String getPython() {
        return preferences.get("python", isWindows() ? "python" : "python3");
    }

    public synchronized void configure(Path serviceDir, Path modelsDir, int port, String python) {
        if (process != null && process.isAlive()) {
            throw new IllegalStateException(Translations.getString("VisionService.StopBeforeConfigure"));
        }
        if (port < 1 || port > 65535 || python.trim().isEmpty()) {
            throw new IllegalArgumentException(Translations.getString("VisionService.InvalidConfiguration"));
        }
        preferences.put("serviceDir", serviceDir.toAbsolutePath().normalize().toString());
        preferences.put("modelsDir", modelsDir.toAbsolutePath().normalize().toString());
        preferences.putInt("port", port);
        preferences.put("python", python.trim());
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).contains("win");
    }

    Path environmentPython() {
        return getServiceDir().resolve(isWindows() ? ".venv/Scripts/python.exe" : ".venv/bin/python");
    }

    public String servicePython() {
        Path python = environmentPython();
        return Files.isRegularFile(python) ? python.toString() : getPython();
    }

    public synchronized void start() throws Exception {
        if (isRunning()) {
            return;
        }
        stop();
        if (!Files.isDirectory(getServiceDir().resolve("vision_service"))
                || !Files.isRegularFile(getModelsDir().resolve("manifest.json"))) {
            throw new IOException(Translations.getString("VisionService.MissingFiles"));
        }
        // Do not mistake a different process already on this port for our own service.
        try (ServerSocket probe = new ServerSocket()) {
            probe.bind(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), getPort()));
        }
        Path log = getServiceDir().resolve("vision-service.log");
        process = new ProcessBuilder(servicePython(), "-m", "vision_service", "--models-dir",
                getModelsDir().toString(), "--port", Integer.toString(getPort()), "--host", "127.0.0.1")
                .directory(getServiceDir().toFile()).redirectErrorStream(true)
                .redirectOutput(ProcessBuilder.Redirect.appendTo(log.toFile())).start();
        try {
            HttpClient client = HttpClient.newBuilder().proxy(ProxySelector.of(null))
                    .connectTimeout(Duration.ofSeconds(1)).build();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
            while (process.isAlive() && System.nanoTime() < deadline) {
                try {
                    HttpResponse<String> response = client.send(HttpRequest.newBuilder(
                            URI.create("http://127.0.0.1:" + getPort() + "/v1/health"))
                            .timeout(Duration.ofSeconds(1)).GET().build(), HttpResponse.BodyHandlers.ofString());
                    if (response.statusCode() == 200 && isBottomVisionReady(response.body())) {
                        ready = true;
                        return;
                    }
                }
                catch (IOException | RuntimeException e) {
                    // Service may still be importing dependencies or loading the encoder.
                }
                Thread.sleep(200);
            }
            throw new IOException(Translations.getString("VisionService.StartFailed") + " " + log);
        }
        catch (Exception e) {
            stop();
            throw e;
        }
    }

    static boolean isBottomVisionReady(String body) {
        JsonObject health = new JsonParser().parse(body).getAsJsonObject();
        return "openpnp-vision".equals(health.get("service").getAsString())
                && health.get("contract_version").getAsInt() == 1
                && health.has("encoder_version") && !health.get("encoder_version").isJsonNull()
                && "ok".equals(health.getAsJsonObject("heads").getAsJsonObject("bottom-vision")
                        .get("status").getAsString());
    }

    public synchronized void stop() {
        ready = false;
        Process current = process;
        process = null;
        if (current != null) {
            current.destroy();
            try {
                if (!current.waitFor(2, TimeUnit.SECONDS)) {
                    current.destroyForcibly();
                }
            }
            catch (InterruptedException e) {
                current.destroyForcibly();
                Thread.currentThread().interrupt();
            }
        }
    }

    public static void warnIfDisabled(Component parent) {
        if (!INSTANCE.isEnabled()) {
            JOptionPane.showMessageDialog(parent, Translations.getString("VisionService.EnableFirst"),
                    Translations.getString("VisionService.Title"), JOptionPane.INFORMATION_MESSAGE);
        }
    }

    /** All process startup and health polling runs off the Swing event thread. */
    public void addMenuItems(JMenu menu, Component parent) {
        JCheckBoxMenuItem enabled = new JCheckBoxMenuItem(Translations.getString("VisionService.Title"), isEnabled());
        menu.add(enabled);
        JMenuItem install = new JMenuItem(Translations.getString("VisionService.Install"));
        menu.add(install);
        install.addActionListener(e -> new VisionServiceInstallationDialog(parent, this).setVisible(true));
        Runnable toggle = () -> {
            boolean requested = enabled.isSelected();
            setEnabled(requested);
            enabled.setEnabled(false);
            install.setEnabled(false);
            new SwingWorker<Void, Void>() {
                @Override
                protected Void doInBackground() throws Exception {
                    if (requested) {
                        start();
                    }
                    else {
                        stop();
                    }
                    return null;
                }
                @Override
                protected void done() {
                    try {
                        get();
                    }
                    catch (Exception error) {
                        setEnabled(false);
                        showError(parent, error);
                    }
                    enabled.setSelected(isEnabled());
                    enabled.setEnabled(true);
                    install.setEnabled(true);
                }
            }.execute();
        };
        enabled.addActionListener(e -> toggle.run());
        Timer state = new Timer(1000, e -> {
            if (enabled.isEnabled()) {
                enabled.setSelected(isEnabled());
            }
            enabled.setText(Translations.getString(isRunning()
                    ? "VisionService.Running" : "VisionService.Title"));
        });
        state.start();
        if (isEnabled()) {
            toggle.run();
        }
    }

    static void showError(Component parent, Exception error) {
        Throwable cause = error.getCause() == null ? error : error.getCause();
        JOptionPane.showMessageDialog(parent, cause.getMessage(),
                Translations.getString("VisionService.Title"), JOptionPane.ERROR_MESSAGE);
    }
}
