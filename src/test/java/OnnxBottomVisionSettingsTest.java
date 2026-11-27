import static org.junit.jupiter.api.Assertions.*;

import java.io.StringWriter;
import java.nio.file.Files;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openpnp.machine.reference.vision.ReferenceBottomVision;
import org.openpnp.machine.reference.vision.wizards.OnnxBottomVisionSettingsConfigurationWizard;
import org.openpnp.model.BottomVisionSettings;
import org.openpnp.model.Configuration;
import org.openpnp.model.Configuration.VisionSettingsConfigurationHolder;
import org.openpnp.model.LengthUnit;
import org.openpnp.model.Location;
import org.openpnp.model.OnnxBottomVisionSettings;
import org.openpnp.util.XmlSerialize;
import org.simpleframework.xml.Serializer;

class OnnxBottomVisionSettingsTest {
    @BeforeEach
    void initialize() throws Exception {
        Configuration.initialize(Files.createTempDirectory("onnx-settings-").toFile());
    }

    @Test
    void copiesSharedValuesAndPreservesEngineWhenSpecializing() throws Exception {
        BottomVisionSettings source = new BottomVisionSettings();
        source.setEnabled(true);
        source.setPreRotateUsage(ReferenceBottomVision.PreRotateUsage.AlwaysOn);
        source.setCheckPartSizeMethod(ReferenceBottomVision.PartSizeCheckMethod.BodySize);
        source.setCheckSizeTolerancePercent(17);
        source.setMaxRotation(ReferenceBottomVision.MaxRotation.Full);
        source.setAsymmetric(true);
        source.setVisionOffset(new Location(LengthUnit.Millimeters, 0.2, 0.3, 0, 0));
        source.setPipelineParameterAssignments(Map.of("unused", 1));
        OnnxBottomVisionSettings settings = new OnnxBottomVisionSettings();
        settings.setValues(source);
        assertTrue(settings.isEnabled());
        assertFalse(settings.usePipeline());
        assertEquals(source.getPreRotateUsage(), settings.getPreRotateUsage());
        assertEquals(source.getCheckPartSizeMethod(), settings.getCheckPartSizeMethod());
        assertEquals(17, settings.getCheckSizeTolerancePercent());
        assertEquals(source.getMaxRotation(), settings.getMaxRotation());
        assertEquals(source.getVisionOffset(), settings.getVisionOffset());
        assertTrue(settings.isAsymmetric());
        assertNull(settings.getPipelineParameterAssignments());
        assertTrue(settings.createSpecialization() instanceof OnnxBottomVisionSettings);
        assertTrue(new BottomVisionSettings().usePipeline());
        assertEquals(BottomVisionSettings.class, new BottomVisionSettings().createSpecialization().getClass());
        assertFalse(serialize(settings).contains("cv-pipeline"));
        settings.resetToDefault();
        assertTrue(settings.isEnabled());
        assertFalse(settings.isAsymmetric());
        assertEquals(0, settings.getVisionOffset().getX());
        assertEquals(ReferenceBottomVision.PreRotateUsage.Default, settings.getPreRotateUsage());
        assertEquals(ReferenceBottomVision.PartSizeCheckMethod.Disabled, settings.getCheckPartSizeMethod());
        assertEquals(ReferenceBottomVision.MaxRotation.Adjust, settings.getMaxRotation());
        assertEquals(20, settings.getCheckSizeTolerancePercent());
    }

    @Test
    void roundTripsAlongsideOpenCvSettingsWithoutSerializingAPipeline() throws Exception {
        OnnxBottomVisionSettings settings = new OnnxBottomVisionSettings();
        settings.setName("ONNX test");
        settings.resetToDefault();
        settings.setCheckSizeTolerancePercent(12);
        settings.setAsymmetric(true);
        settings.setVisionOffset(new Location(LengthUnit.Millimeters, 0.1, -0.3, 0, 0));
        String xml = serialize(settings);
        assertFalse(xml.contains("cvPipeline"), xml);
        Serializer serializer = XmlSerialize.createSerializer();
        VisionSettingsConfigurationHolder restored = serializer.read(VisionSettingsConfigurationHolder.class, xml);
        assertEquals(OnnxBottomVisionSettings.class, restored.visionSettings.get(0).getClass());
        OnnxBottomVisionSettings copy = (OnnxBottomVisionSettings) restored.visionSettings.get(0);
        assertEquals(settings.getId(), copy.getId());
        assertEquals("ONNX test", copy.getName());
        assertEquals(12, copy.getCheckSizeTolerancePercent());
        assertEquals(settings.getVisionOffset(), copy.getVisionOffset());
        assertTrue(copy.isAsymmetric());
        // The base type must still preserve its actual OpenCV pipeline.
        BottomVisionSettings classic = new BottomVisionSettings();
        assertTrue(serialize(classic).contains("cv-pipeline"));
    }

    @Test
    void createsTypeAwareWizardsForSettingsAndPartAlignment() throws Exception {
        Configuration.get().load();
        OnnxBottomVisionSettings settings = new OnnxBottomVisionSettings();
        settings.resetToDefault();
        settings.setName("ONNX wizard test");
        Configuration.get().addVisionSettings(settings);
        ReferenceBottomVision alignment = ReferenceBottomVision.getDefault();
        alignment.setBottomVisionSettings(settings);
        assertTrue(settings.getConfigurationWizard() instanceof OnnxBottomVisionSettingsConfigurationWizard);
        assertFalse(hasPipelineControls(settings.getConfigurationWizard().getWizardPanel()));
        assertTrue(hasPipelineControls(new BottomVisionSettings().getConfigurationWizard().getWizardPanel()));
        assertTrue(alignment.getPartConfigurationWizard(alignment) instanceof OnnxBottomVisionSettingsConfigurationWizard);
        assertEquals(2, alignment.getPropertySheets().length);
    }

    @Test
    void readsConfigurationsSavedWithRetiredObserverAttributes() throws Exception {
        ReferenceBottomVision alignment = new ReferenceBottomVision();
        StringWriter xml = new StringWriter();
        Serializer serializer = XmlSerialize.createSerializer();
        serializer.write(alignment, xml);
        String legacy = xml.toString().replaceFirst("<reference-bottom-vision",
                "<reference-bottom-vision shadow-mode=\"true\" shadow-port=\"8010\" shadow-timeout-ms=\"1000\"");
        assertTrue(legacy.contains("shadow-mode="));
        ReferenceBottomVision restored = serializer.read(ReferenceBottomVision.class, legacy);
        StringWriter saved = new StringWriter();
        serializer.write(restored, saved);
        assertFalse(saved.toString().contains("shadow-"));
    }

    private boolean hasPipelineControls(java.awt.Container container) {
        for (java.awt.Component child : container.getComponents()) {
            if (child instanceof org.openpnp.gui.components.PipelineControls
                    || child instanceof java.awt.Container && hasPipelineControls((java.awt.Container) child)) {
                return true;
            }
        }
        return false;
    }

    private String serialize(BottomVisionSettings settings) throws Exception {
        VisionSettingsConfigurationHolder holder = new VisionSettingsConfigurationHolder();
        holder.visionSettings.add(settings);
        StringWriter xml = new StringWriter();
        XmlSerialize.createSerializer().write(holder, xml);
        return xml.toString();
    }
}
