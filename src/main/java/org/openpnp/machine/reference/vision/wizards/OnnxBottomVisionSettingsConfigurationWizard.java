package org.openpnp.machine.reference.vision.wizards;

import org.openpnp.Translations;
import org.openpnp.model.OnnxBottomVisionSettings;
import org.openpnp.model.PartSettingsHolder;

/**
 * Wizard for ONNX-engine Bottom Vision settings. The pipeline section of the base wizard is
 * hidden automatically because ONNX settings do not use an OpenCV pipeline.
 */
public class OnnxBottomVisionSettingsConfigurationWizard extends BottomVisionSettingsConfigurationWizard {

    public OnnxBottomVisionSettingsConfigurationWizard(OnnxBottomVisionSettings visionSettings,
            PartSettingsHolder settingsHolder) {
        super(visionSettings, settingsHolder);
    }

    @Override
    public String getWizardName() {
        return Translations.getString("OnnxBottomVisionSettingsConfigurationWizard.wizardName"); //$NON-NLS-1$
    }
}
