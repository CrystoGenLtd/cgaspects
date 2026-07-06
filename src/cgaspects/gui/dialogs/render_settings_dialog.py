"""Non-modal dialog for sphere material, lighting and ambient-occlusion settings.

Emits a fresh :class:`RenderSettings` on every change; the main window forwards
it to ``VisualisationWidget.set_render_settings`` so edits preview live.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from ..visualisation.shading import MATERIAL_PRESETS, RenderSettings


class RenderSettingsDialog(QDialog):
    settingsChanged = Signal(object)  # RenderSettings

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sphere & Lighting Settings")
        self.setModal(False)
        self.resize(420, 520)

        self._updating = False  # guard against feedback while applying presets

        main_layout = QVBoxLayout(self)

        # ------------------------------------------------------------ material
        material_group = QGroupBox("Material")
        material_layout = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(MATERIAL_PRESETS) + ["Custom"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        material_layout.addRow("Preset:", self.preset_combo)

        self.ambient_spin = self._float_spin(0.0, 1.0, 0.05)
        material_layout.addRow("Ambient:", self.ambient_spin)

        self.diffuse_spin = self._float_spin(0.0, 1.0, 0.05)
        material_layout.addRow("Diffuse:", self.diffuse_spin)

        self.specular_spin = self._float_spin(0.0, 1.0, 0.05)
        material_layout.addRow("Specular:", self.specular_spin)

        self.shininess_spin = self._float_spin(1.0, 128.0, 4.0, decimals=0)
        material_layout.addRow("Shininess:", self.shininess_spin)

        self.tint_spin = self._float_spin(0.0, 1.0, 0.05)
        self.tint_spin.setToolTip(
            "0 = white highlights (plastic), 1 = highlights tinted by the sphere colour (metal)"
        )
        material_layout.addRow("Specular Tint:", self.tint_spin)

        self.toon_spin = QSpinBox()
        self.toon_spin.setRange(0, 8)
        self.toon_spin.setSpecialValueText("Off")
        self.toon_spin.setToolTip("Number of cel-shading bands (0 = smooth shading)")
        self.toon_spin.valueChanged.connect(self._on_field_edited)
        material_layout.addRow("Toon Bands:", self.toon_spin)

        material_group.setLayout(material_layout)
        main_layout.addWidget(material_group)

        # ------------------------------------------------------------ lighting
        light_group = QGroupBox("Lighting (follows camera)")
        light_layout = QFormLayout()

        self.azimuth_slider, azimuth_row = self._angle_slider(-180, 180)
        light_layout.addRow("Azimuth:", azimuth_row)

        self.elevation_slider, elevation_row = self._angle_slider(-90, 90)
        light_layout.addRow("Elevation:", elevation_row)

        light_group.setLayout(light_layout)
        main_layout.addWidget(light_group)

        # ------------------------------------------------- ambient occlusion
        ao_group = QGroupBox("Ambient Occlusion")
        ao_layout = QFormLayout()

        self.ao_checkbox = QCheckBox()
        self.ao_checkbox.setToolTip(
            "Darken particles buried inside the crystal based on neighbour density.\n"
            "Recomputed when data loads; may take a few seconds on large point sets."
        )
        self.ao_checkbox.toggled.connect(self._on_field_edited)
        ao_layout.addRow("Enable:", self.ao_checkbox)

        self.ao_strength_spin = self._float_spin(0.0, 1.0, 0.05)
        ao_layout.addRow("Strength:", self.ao_strength_spin)

        ao_group.setLayout(ao_layout)
        main_layout.addWidget(ao_group)

        # ------------------------------------------------------------- buttons
        button_layout = QHBoxLayout()
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        main_layout.addLayout(button_layout)
        main_layout.addStretch()

        self._apply_to_widgets(RenderSettings())

    # ------------------------------------------------------------------

    def _float_spin(self, lo, hi, step, decimals=2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(self._on_field_edited)
        return spin

    def _angle_slider(self, lo, hi):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        label = QLabel()
        label.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v: label.setText(f"{v}°"))
        slider.valueChanged.connect(self._on_light_changed)
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(label)
        return slider, row

    # ------------------------------------------------------------------

    def current_settings(self) -> RenderSettings:
        return RenderSettings(
            material=self.preset_combo.currentText(),
            ambient=self.ambient_spin.value(),
            diffuse=self.diffuse_spin.value(),
            specular=self.specular_spin.value(),
            shininess=self.shininess_spin.value(),
            specular_tint=self.tint_spin.value(),
            toon_levels=self.toon_spin.value(),
            light_azimuth=float(self.azimuth_slider.value()),
            light_elevation=float(self.elevation_slider.value()),
            ao_enabled=self.ao_checkbox.isChecked(),
            ao_strength=self.ao_strength_spin.value(),
        )

    def _apply_to_widgets(self, settings: RenderSettings):
        self._updating = True
        try:
            self.preset_combo.setCurrentText(settings.material)
            self.ambient_spin.setValue(settings.ambient)
            self.diffuse_spin.setValue(settings.diffuse)
            self.specular_spin.setValue(settings.specular)
            self.shininess_spin.setValue(settings.shininess)
            self.tint_spin.setValue(settings.specular_tint)
            self.toon_spin.setValue(settings.toon_levels)
            self.azimuth_slider.setValue(round(settings.light_azimuth))
            self.elevation_slider.setValue(round(settings.light_elevation))
            self.ao_checkbox.setChecked(settings.ao_enabled)
            self.ao_strength_spin.setValue(settings.ao_strength)
        finally:
            self._updating = False

    # ------------------------------------------------------------------

    def _on_preset_changed(self, name: str):
        if self._updating or name not in MATERIAL_PRESETS:
            return
        settings = self.current_settings().with_preset(name)
        self._apply_to_widgets(settings)
        self.settingsChanged.emit(settings)

    def _on_field_edited(self, *_):
        if self._updating:
            return
        # Manual edits to material fields switch the preset to Custom
        sender = self.sender()
        if sender not in (self.ao_checkbox, self.ao_strength_spin):
            self._updating = True
            self.preset_combo.setCurrentText("Custom")
            self._updating = False
        self.settingsChanged.emit(self.current_settings())

    def _on_light_changed(self, *_):
        if self._updating:
            return
        self.settingsChanged.emit(self.current_settings())

    def _reset_defaults(self):
        settings = RenderSettings()
        self._apply_to_widgets(settings)
        self.settingsChanged.emit(settings)
