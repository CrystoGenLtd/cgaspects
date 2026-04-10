import numpy as np
from PySide6.QtGui import QMatrix4x4, QQuaternion, QVector3D, QVector4D
import logging

LOG = logging.getLogger("CGA:Camera")


def pca(vertices):
    "returns the magnitudes and the principle axes"
    if len(vertices) < 2:
        return np.ones(3), np.eye(3)
    centered_vertices = vertices - np.mean(vertices, axis=0)
    covariance_matrix = np.cov(centered_vertices.T)
    try:
        U, S, vh = np.linalg.svd(covariance_matrix)
    except np.linalg.LinAlgError:
        return np.ones(3), np.eye(3)
    S = np.where(S > 0, S, 1.0)
    return S, vh


def bounding_box(vertices):
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


class Camera:
    def __init__(self, position=QVector3D(0, 0, -20), target=QVector3D(0, 0, 0)):
        self.position = position
        self.target = target
        self.up = QVector3D(0, 1, 0)
        self.right = QVector3D(1, 0, 0)
        self.screen = QVector3D(0, 0, 1)

        self.rotationSpeed = 1.0
        self.zoomSpeed = 0.02
        self.scale = 1.0
        self.model_rotation = QQuaternion()  # identity = no object rotation
        self.distance = (self.position - self.target).length()
        self.storeOrientation()

        self.perspectiveProjection = True
        self.fieldOfView = 45
        self.orthoSize = 10
        self.nearPlane = 0.01
        self.farPlane = 1000

    def modelMatrix(self):
        model = QMatrix4x4()
        model.scale(self.scale, self.scale, self.scale)
        model.rotate(self.model_rotation)
        return model

    def modelRotationMatrix(self):
        """Pure rotation matrix for the object (no scale), used by axes renderer."""
        mat = QMatrix4x4()
        mat.rotate(self.model_rotation)
        return mat

    def viewMatrix(self):
        view = QMatrix4x4()
        view.lookAt(self.position, self.target, self.up)
        return view

    def modelViewMatrix(self):
        return self.viewMatrix() * self.modelMatrix()

    def projectionMatrix(self, aspectRatio):
        projection = QMatrix4x4()
        if self.perspectiveProjection:
            projection.perspective(
                self.fieldOfView, aspectRatio, self.nearPlane, self.farPlane
            )
        else:
            halfWidth = aspectRatio * self.orthoSize
            halfHeight = self.orthoSize
            projection.ortho(
                -halfWidth,
                halfWidth,
                -halfHeight,
                halfHeight,
                self.nearPlane,
                self.farPlane,
            )
        return projection

    def modelViewProjectionMatrix(self, aspectRatio):
        return (
            self.projectionMatrix(aspectRatio) * self.viewMatrix() * self.modelMatrix()
        )

    def storeOrientation(self):
        self._stored_orientation = (
            self.position,
            self.target,
            self.right,
            self.up,
            self.scale,
            QQuaternion(self.model_rotation),
        )

    def resetOrientation(self):
        (
            self.position,
            self.target,
            self.right,
            self.up,
            self.scale,
            self.model_rotation,
        ) = self._stored_orientation

    def rotate_model(self, dx: float, dy: float) -> None:
        """Accumulate object rotation from mouse delta (non-destructive)."""
        q_yaw = QQuaternion.fromAxisAndAngle(self.up, dx * self.rotationSpeed)
        q_pitch = QQuaternion.fromAxisAndAngle(self.right, -dy * self.rotationSpeed)
        self.model_rotation = (q_yaw * q_pitch * self.model_rotation).normalized()

    def reset_model_rotation(self) -> None:
        """Reset the object rotation to identity."""
        self.model_rotation = QQuaternion()

    def compute_relative_mouse_pos(self, event_pos):
        mouse_x = (self.right * event_pos.windowPos().x()).normalized()
        mouse_y = (self.up * event_pos.windowPos().y()).normalized()
        return (mouse_x + mouse_y).normalized()

    def orbit(self, dx, dy, restrict_axis=None, event_pos=QVector3D(0, 0, 0)):
        # Create quaternions representing the rotations
        q_pitch = QQuaternion.fromAxisAndAngle(self.right, dy * self.rotationSpeed)
        q_yaw = QQuaternion.fromAxisAndAngle(self.up, -dx * self.rotationSpeed)

        # Apply axis restriction
        if restrict_axis == "shift_x":
            q_yaw = QQuaternion()
        elif restrict_axis == "shift_y":
            q_pitch = QQuaternion()
        elif restrict_axis == "shift_z":
            q_yaw = QQuaternion()
            q_pitch = QQuaternion()

            # Calculate roll magnitude based on tangential movement
            # Cross product of position vector and movement vector gives roll direction
            pos_vec = QVector3D(event_pos.x(), event_pos.y(), 0).normalized()
            move_vec = QVector3D(dx, dy, 0).normalized()
            roll_direction = QVector3D.crossProduct(pos_vec, move_vec).z()
            roll_magnitude = np.sqrt(dx**2 + dy**2) * self.rotationSpeed
            q_roll = QQuaternion.fromAxisAndAngle(
                self.screen, -roll_direction * roll_magnitude
            )

        # Rotate the position around the target
        direction = self.position - self.target
        direction = q_pitch.rotatedVector(direction)
        direction = q_yaw.rotatedVector(direction)
        self.position = self.target + direction

        # Update view vectors
        self.screen = (self.target - self.position).normalized()
        self.right = QVector3D.crossProduct(self.up, self.screen).normalized()

        if restrict_axis == "shift_z":
            self.right = q_roll.rotatedVector(self.right)
        self.up = QVector3D.crossProduct(self.screen, self.right).normalized()

        # Recalculate the distance in case of zooming
        self.distance = (self.position - self.target).length()

    def zoom(self, amount):
        self.scale *= 1 + amount * self.zoomSpeed
        if amount > 0:
            self.scale *= 1 + self.zoomSpeed
        elif amount < 0:
            self.scale /= 1 + self.zoomSpeed

    def updatePosition(self):
        direction = self.position - self.target
        direction = direction / direction.length()
        self.position = self.target + direction * self.distance

    def pan(self, dx, dy):
        # Adjust these factors to change the sensitivity of panning
        panSpeed = 0.1
        right = self.front.cross(self.front, self.up)
        up = self.right.cross(right, self.front)

        self.target += right * dx * panSpeed
        self.target += up * dy * panSpeed

    def toggleProjectionMode(self):
        self.perspectiveProjection = not self.perspectiveProjection

    def setProjectionMode(self, kind):
        if kind.lower() == "orthographic":
            self.perspectiveProjection = False
        elif kind.lower() == "perspective":
            self.perspectiveProjection = True
        else:
            LOG.error("unknown projection mode", kind)

    def fitToObject(self, points):
        extents, axes = pca(points)
        self.right = QVector3D(axes[0, 0], axes[0, 1], axes[0, 2])
        self.up = QVector3D.crossProduct(self.target - self.position, self.right)
        self.scale = 1 / np.sqrt(np.max(extents))
        self.storeOrientation()

    def projectionMode(self):
        return "Perspective" if self.perspectiveProjection else "Orthographic"

    def snapshot(self) -> "CameraSnapshot":
        """Return a CameraSnapshot capturing the full animatable camera + object state."""
        from ..animation.keyframe import CameraSnapshot
        return CameraSnapshot(
            position=QVector3D(self.position),
            target=QVector3D(self.target),
            up=QVector3D(self.up),
            scale=self.scale,
            perspective=self.perspectiveProjection,
            model_rotation=QQuaternion(self.model_rotation),
        )

    def restore_snapshot(self, snap: "CameraSnapshot") -> None:
        """Apply a CameraSnapshot, recomputing derived camera vectors."""
        self.position = QVector3D(snap.position)
        self.target = QVector3D(snap.target)
        self.up = QVector3D(snap.up)
        self.scale = snap.scale
        self.perspectiveProjection = snap.perspective
        self.model_rotation = QQuaternion(snap.model_rotation)
        # Recompute derived vectors
        self.screen = (self.target - self.position).normalized()
        self.right = QVector3D.crossProduct(self.up, self.screen).normalized()
        self.up = QVector3D.crossProduct(self.screen, self.right).normalized()
        self.distance = (self.position - self.target).length()
