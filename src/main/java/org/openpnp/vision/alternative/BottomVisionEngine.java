package org.openpnp.vision.alternative;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Analyzes one captured frame independently of the camera, machine and CvPipeline.
 */
public interface BottomVisionEngine {
    Result analyze(BufferedImage image) throws IOException, InterruptedException;

    enum Status {
        DETECTED, NO_DETECTION, UNCERTAIN, ERROR
    }

    /**
     * Coordinates and dimensions refer to the supplied image, with the origin at the top left.
     * The angle is clockwise in degrees along the first side, corresponding to width.
     * The caller applies part symmetry and the camera-to-machine transformation.
     */
    final class Pose {
        public final double x;
        public final double y;
        public final double width;
        public final double height;
        public final double angle;

        public Pose(double x, double y, double width, double height, double angle) {
            if (!Double.isFinite(x) || !Double.isFinite(y) || x < 0 || y < 0
                    || !Double.isFinite(width) || !Double.isFinite(height) || width <= 0 || height <= 0
                    || !Double.isFinite(angle)) {
                throw new IllegalArgumentException("Invalid image-space pose");
            }
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.angle = angle;
        }
    }

    final class Result {
        private final Status status;
        private final Pose pose;
        private final Double confidence;

        public Result(Status status, Pose pose, Double confidence) {
            this.status = Objects.requireNonNull(status, "status");
            if ((status == Status.DETECTED) != (pose != null)) {
                throw new IllegalArgumentException("Only a detected result can contain a pose");
            }
            if (confidence != null && (!Double.isFinite(confidence) || confidence < 0 || confidence > 1)) {
                throw new IllegalArgumentException("Confidence must be between zero and one");
            }
            this.pose = pose;
            this.confidence = confidence;
        }

        public Status getStatus() {
            return status;
        }

        public Optional<Pose> getPose() {
            return Optional.ofNullable(pose);
        }

        public OptionalDouble getConfidence() {
            return confidence == null ? OptionalDouble.empty() : OptionalDouble.of(confidence);
        }
    }
}
