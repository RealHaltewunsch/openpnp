package org.openpnp.vision.service;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dialog;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;

import org.openpnp.Translations;

/** Installs dependencies into a service-local virtual environment without changing system Python. */
public class VisionServiceInstallationDialog extends JDialog {
    private final VisionServiceController controller;
    private final JTextField serviceDir;
    private final JTextField modelsDir;
    private final JTextField python;
    private final JSpinner port;
    private final JTextArea log = new JTextArea(12, 75);
    private final JButton install = new JButton(text("VisionService.InstallDependencies"));
    private final JButton start = new JButton(text("VisionService.Start"));
    private final JButton close = new JButton(text("VisionService.Close"));
    private final JPanel fields = new JPanel(new GridLayout(0, 1, 4, 4));

    public VisionServiceInstallationDialog(Component parent, VisionServiceController controller) {
        super(SwingUtilities.getWindowAncestor(parent), text("VisionService.Install"), Dialog.ModalityType.APPLICATION_MODAL);
        this.controller = controller;
        serviceDir = new JTextField(controller.getServiceDir().toString(), 40);
        modelsDir = new JTextField(controller.getModelsDir().toString(), 40);
        python = new JTextField(controller.getPython(), 40);
        port = new JSpinner(new SpinnerNumberModel(controller.getPort(), 1, 65535, 1));
        JPanel content = new JPanel(new BorderLayout(8, 8));
        content.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        setContentPane(content);
        JPanel top = new JPanel(new BorderLayout(8, 8));
        top.add(new JLabel(text("VisionService.InstallExplanation")), BorderLayout.NORTH);
        addPathField("VisionService.ServiceDirectory", serviceDir);
        addPathField("VisionService.ModelsDirectory", modelsDir);
        fields.add(new JLabel(text("VisionService.Python")));
        fields.add(python);
        fields.add(new JLabel(text("VisionService.Port")));
        fields.add(port);
        top.add(fields, BorderLayout.CENTER);
        content.add(top, BorderLayout.NORTH);
        log.setEditable(false);
        content.add(new JScrollPane(log), BorderLayout.CENTER);
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttons.add(install);
        buttons.add(start);
        close.addActionListener(e -> dispose());
        buttons.add(close);
        content.add(buttons, BorderLayout.SOUTH);
        install.addActionListener(e -> runTask(true));
        start.addActionListener(e -> runTask(false));
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        pack();
        setLocationRelativeTo(parent);
    }

    private static String text(String key) {
        return Translations.getString(key);
    }

    private void addPathField(String key, JTextField field) {
        fields.add(new JLabel(text(key)));
        JPanel row = new JPanel(new BorderLayout(5, 0));
        row.add(field, BorderLayout.CENTER);
        JButton browse = new JButton(text("VisionService.Browse"));
        browse.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser(field.getText());
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                String oldDefault = Path.of(serviceDir.getText()).resolve("models/synthetic").toString();
                field.setText(chooser.getSelectedFile().getAbsolutePath());
                if (field == serviceDir && modelsDir.getText().equals(oldDefault)) {
                    modelsDir.setText(Path.of(serviceDir.getText()).resolve("models/synthetic").toString());
                }
            }
        });
        row.add(browse, BorderLayout.EAST);
        fields.add(row);
    }

    private void runTask(boolean installing) {
        try {
            controller.configure(Path.of(serviceDir.getText()), Path.of(modelsDir.getText()),
                    (Integer) port.getValue(), python.getText());
        }
        catch (Exception e) {
            VisionServiceController.showError(this, e);
            return;
        }
        close.setEnabled(false);
        install.setEnabled(false);
        start.setEnabled(false);
        setFieldsEnabled(fields, false);
        // Keep the log visible until the operation completes; closing cannot orphan an installer.
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                if (installing) {
                    execute(List.of(controller.getPython(), "-m", "venv",
                            controller.getServiceDir().resolve(".venv").toString()));
                    execute(List.of(controller.environmentPython().toString(), "-m", "pip", "install",
                            "onnxruntime", "numpy", "Pillow", "onnx"));
                    publish(text("VisionService.InstallComplete"));
                }
                else {
                    controller.start();
                    controller.setEnabled(true);
                    publish(text("VisionService.Running"));
                }
                return null;
            }

            private void execute(List<String> command) throws Exception {
                Process process = new ProcessBuilder(command).directory(controller.getServiceDir().toFile())
                        .redirectErrorStream(true).start();
                try (BufferedReader output = new BufferedReader(new InputStreamReader(
                        process.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = output.readLine()) != null) {
                        publish(line);
                    }
                    if (process.waitFor() != 0) {
                        throw new java.io.IOException(text("VisionService.InstallFailed"));
                    }
                }
                finally {
                    if (process.isAlive()) {
                        process.destroyForcibly();
                    }
                }
            }

            @Override
            protected void process(List<String> lines) {
                for (String line : lines) {
                    log.append(line + "\n");
                }
                log.setCaretPosition(log.getDocument().getLength());
            }

            @Override
            protected void done() {
                try {
                    get();
                }
                catch (Exception e) {
                    VisionServiceController.showError(VisionServiceInstallationDialog.this, e);
                }
                close.setEnabled(true);
                install.setEnabled(true);
                start.setEnabled(true);
                setFieldsEnabled(fields, true);
                setDefaultCloseOperation(DISPOSE_ON_CLOSE);
            }
        }.execute();
    }

    private static void setFieldsEnabled(java.awt.Container parent, boolean enabled) {
        for (Component component : parent.getComponents()) {
            component.setEnabled(enabled);
            if (component instanceof java.awt.Container) {
                setFieldsEnabled((java.awt.Container) component, enabled);
            }
        }
    }
}
