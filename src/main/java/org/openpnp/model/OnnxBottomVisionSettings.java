package org.openpnp.model;

import org.openpnp.gui.support.Wizard;
import org.openpnp.machine.reference.vision.ReferenceBottomVision;
import org.openpnp.machine.reference.vision.wizards.OnnxBottomVisionSettingsConfigurationWizard;

/**
 * Bottom Vision settings for the ONNX Runtime vision engine. The engine runs in a separate
 * Python service process and never uses an OpenCV pipeline: all pipeline state is ignored so
 * the OpenCV pipeline is not loaded or serialized for these settings.
 */
public class OnnxBottomVisionSettings extends BottomVisionSettings {

    public OnnxBottomVisionSettings() {
        this(Configuration.createId("OBV"));
    }

    public OnnxBottomVisionSettings(String id) {
        super(id);
        setPipeline(null);
    }

    @Override
    public boolean usePipeline() {
        return false;
    }

    @Override
    public Wizard getConfigurationWizard() {
        return new OnnxBottomVisionSettingsConfigurationWizard(this, null);
    }

    @Override
    public Wizard getConfigurationWizard(PartSettingsHolder settingsHolder) {
        return new OnnxBottomVisionSettingsConfigurationWizard(this, settingsHolder);
    }

    @Override
    public BottomVisionSettings createSpecialization() {
        return new OnnxBottomVisionSettings();
    }

    @Override
    public void setValues(BottomVisionSettings another) {
        // Copy alignment options without pipeline parameters or stages.
        setEnabled(another.isEnabled());
        setPipeline(null);
        setPipelineParameterAssignments(null);
        setPreRotateUsage(another.getPreRotateUsage());
        setCheckPartSizeMethod(another.getCheckPartSizeMethod());
        setMaxRotation(another.getMaxRotation());
        setCheckSizeTolerancePercent(another.getCheckSizeTolerancePercent());
        setVisionOffset(another.getVisionOffset());
        setAsymmetric(another.isAsymmetric());
        Configuration.get().fireVisionSettingsChanged();
    }

    @Override
    public void resetToDefault() {
        setPipeline(null);
        setPipelineParameterAssignments(null);
        setEnabled(true);
        setPreRotateUsage(ReferenceBottomVision.PreRotateUsage.Default);
        setCheckPartSizeMethod(ReferenceBottomVision.PartSizeCheckMethod.Disabled);
        setCheckSizeTolerancePercent(20);
        setMaxRotation(ReferenceBottomVision.MaxRotation.Adjust);
        setAsymmetric(false);
        Configuration.get().fireVisionSettingsChanged();
    }

}
