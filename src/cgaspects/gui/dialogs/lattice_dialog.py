"""
Dialog for entering lattice parameters to convert Cartesian axes to fractional coordinates.
"""
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QTabWidget,
    QWidget,
    QFileDialog,
    QMessageBox,
)

from ...gui.utils.crystallography import Cell, Crystallography

logger = logging.getLogger("CA:LatticeDialog")


class LatticeParametersDialog(QDialog):
    """Dialog for entering lattice parameters either manually or from a CrystoGen file."""

    parametersAccepted = Signal(Cell)

    def __init__(self, parent=None, cell: Optional[Cell] = None):
        super().__init__(parent)
        self.setWindowTitle("Lattice Parameters")
        self.setModal(True)
        self.resize(500, 300)

        self.cell = cell
        self.setupUI()

        # Pre-fill with existing cell parameters if provided
        if cell is not None:
            self.set_cell_values(cell)

    def setupUI(self):
        """Setup the user interface."""
        layout = QVBoxLayout()

        # Create tab widget for manual entry vs file import
        self.tab_widget = QTabWidget()

        # Manual entry tab
        manual_widget = QWidget()
        manual_layout = QFormLayout()

        self.a_edit = QLineEdit()
        self.a_edit.setPlaceholderText("e.g., 3.6295")
        self.b_edit = QLineEdit()
        self.b_edit.setPlaceholderText("e.g., 9.8128")
        self.c_edit = QLineEdit()
        self.c_edit.setPlaceholderText("e.g., 18.4269")
        self.alpha_edit = QLineEdit()
        self.alpha_edit.setPlaceholderText("e.g., 90.000")
        self.beta_edit = QLineEdit()
        self.beta_edit.setPlaceholderText("e.g., 90.000")
        self.gamma_edit = QLineEdit()
        self.gamma_edit.setPlaceholderText("e.g., 118.042")

        manual_layout.addRow("a (Å):", self.a_edit)
        manual_layout.addRow("b (Å):", self.b_edit)
        manual_layout.addRow("c (Å):", self.c_edit)
        manual_layout.addRow("α (°):", self.alpha_edit)
        manual_layout.addRow("β (°):", self.beta_edit)
        manual_layout.addRow("γ (°):", self.gamma_edit)

        manual_widget.setLayout(manual_layout)
        self.tab_widget.addTab(manual_widget, "Manual Entry")

        # File import tab
        file_widget = QWidget()
        file_layout = QVBoxLayout()

        file_info_label = QLabel(
            "Select a structure file (CIF or CrystoGen format).\n"
            "Lattice parameters will be extracted automatically."
        )
        file_info_label.setWordWrap(True)

        file_button_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Path to CIF or CrystoGen Structure File...")
        self.file_path_edit.setReadOnly(True)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)

        file_button_layout.addWidget(self.file_path_edit)
        file_button_layout.addWidget(self.browse_button)

        self.file_info_display = QLabel("")
        self.file_info_display.setWordWrap(True)

        file_layout.addWidget(file_info_label)
        file_layout.addLayout(file_button_layout)
        file_layout.addWidget(self.file_info_display)
        file_layout.addStretch()

        file_widget.setLayout(file_layout)
        self.tab_widget.addTab(file_widget, "From File")

        layout.addWidget(self.tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept_parameters)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def set_cell_values(self, cell: Cell):
        """Populate the input fields with values from a Cell object."""
        self.a_edit.setText(str(cell.a))
        self.b_edit.setText(str(cell.b))
        self.c_edit.setText(str(cell.c))
        self.alpha_edit.setText(str(cell.alpha))
        self.beta_edit.setText(str(cell.beta))
        self.gamma_edit.setText(str(cell.gamma))

    def browse_file(self):
        """Open file dialog to select CrystoGen structure file or CIF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Structure File",
            "",
            "All Files (*);;CIF Files (*.cif);;CrystoGen Files (*)"
        )

        if file_path:
            self.file_path_edit.setText(file_path)
            # Auto-detect file type and parse accordingly
            if file_path.lower().endswith('.cif'):
                self.parse_cif_file(file_path)
            else:
                self.parse_structure_file(file_path)

    def parse_structure_file(self, file_path: str):
        """
        Parse CrystoGen structure file to extract lattice parameters.

        The file format has a section starting with 'Non primitive data' followed by
        lattice parameters:
          a   b   c
         alpha beta gamma
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Find the "Non primitive data" section
            non_primitive_index = None
            for i, line in enumerate(lines):
                if "Non primitive data" in line:
                    non_primitive_index = i
                    break

            if non_primitive_index is None:
                raise ValueError("Could not find 'Non primitive data' section in file")

            # The next line should contain a, b, c
            abc_line = lines[non_primitive_index + 1].strip()
            a, b, c = map(float, abc_line.split())

            # The line after that should contain alpha, beta, gamma
            angles_line = lines[non_primitive_index + 2].strip()
            alpha, beta, gamma = map(float, angles_line.split())

            # Update the manual entry fields
            self.a_edit.setText(str(a))
            self.b_edit.setText(str(b))
            self.c_edit.setText(str(c))
            self.alpha_edit.setText(str(alpha))
            self.beta_edit.setText(str(beta))
            self.gamma_edit.setText(str(gamma))

            # Display info
            info_text = (
                f"Lattice parameters extracted:\n"
                f"a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å\n"
                f"α = {alpha:.3f}°, β = {beta:.3f}°, γ = {gamma:.3f}°"
            )
            self.file_info_display.setText(info_text)
            self.file_info_display.setStyleSheet("color: green;")

            logger.info(f"Parsed lattice parameters from {file_path}")

        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            self.file_info_display.setText(error_msg)
            self.file_info_display.setStyleSheet("color: red;")
            logger.error(error_msg)
            QMessageBox.warning(self, "Parse Error", error_msg)

    def parse_cif_file(self, file_path: str):
        """
        Parse CIF file to extract lattice parameters.

        Looks for the following CIF tags:
        _cell_length_a, _cell_length_b, _cell_length_c
        _cell_angle_alpha, _cell_angle_beta, _cell_angle_gamma
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Initialize parameters
            a = b = c = None
            alpha = beta = gamma = None

            # Parse CIF file line by line
            for line in lines:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Split on whitespace
                parts = line.split()
                if len(parts) < 2:
                    continue

                # Extract lattice parameters
                if parts[0] == '_cell_length_a':
                    a = self._parse_cif_value(parts[1])
                elif parts[0] == '_cell_length_b':
                    b = self._parse_cif_value(parts[1])
                elif parts[0] == '_cell_length_c':
                    c = self._parse_cif_value(parts[1])
                elif parts[0] == '_cell_angle_alpha':
                    alpha = self._parse_cif_value(parts[1])
                elif parts[0] == '_cell_angle_beta':
                    beta = self._parse_cif_value(parts[1])
                elif parts[0] == '_cell_angle_gamma':
                    gamma = self._parse_cif_value(parts[1])

            # Check if all parameters were found
            if None in [a, b, c, alpha, beta, gamma]:
                missing = []
                if a is None: missing.append('a')
                if b is None: missing.append('b')
                if c is None: missing.append('c')
                if alpha is None: missing.append('alpha')
                if beta is None: missing.append('beta')
                if gamma is None: missing.append('gamma')
                raise ValueError(f"Missing lattice parameters in CIF file: {', '.join(missing)}")

            # Update the manual entry fields
            self.a_edit.setText(str(a))
            self.b_edit.setText(str(b))
            self.c_edit.setText(str(c))
            self.alpha_edit.setText(str(alpha))
            self.beta_edit.setText(str(beta))
            self.gamma_edit.setText(str(gamma))

            # Display info
            info_text = (
                f"Lattice parameters extracted from CIF:\n"
                f"a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å\n"
                f"α = {alpha:.3f}°, β = {beta:.3f}°, γ = {gamma:.3f}°"
            )
            self.file_info_display.setText(info_text)
            self.file_info_display.setStyleSheet("color: green;")

            logger.info(f"Parsed lattice parameters from CIF file: {file_path}")

        except Exception as e:
            error_msg = f"Error parsing CIF file: {str(e)}"
            self.file_info_display.setText(error_msg)
            self.file_info_display.setStyleSheet("color: red;")
            logger.error(error_msg)
            QMessageBox.warning(self, "Parse Error", error_msg)

    def _parse_cif_value(self, value_str: str) -> float:
        """
        Parse a CIF value, handling uncertainty notation.

        CIF values may include uncertainty in parentheses, e.g., "13.532(1)"
        This method extracts just the numeric value.
        """
        # Remove uncertainty notation if present (e.g., "13.532(1)" -> "13.532")
        if '(' in value_str:
            value_str = value_str.split('(')[0]

        return float(value_str)

    def accept_parameters(self):
        """Validate and accept the entered parameters."""
        try:
            a = float(self.a_edit.text())
            b = float(self.b_edit.text())
            c = float(self.c_edit.text())
            alpha = float(self.alpha_edit.text())
            beta = float(self.beta_edit.text())
            gamma = float(self.gamma_edit.text())

            # Basic validation
            if any(val <= 0 for val in [a, b, c]):
                raise ValueError("Lattice parameters a, b, c must be positive")

            if not all(0 < angle < 180 for angle in [alpha, beta, gamma]):
                raise ValueError("Angles must be between 0 and 180 degrees")

            # Create Cell object
            self.cell = Cell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

            # Emit signal and accept dialog
            self.parametersAccepted.emit(self.cell)
            self.accept()

            logger.info(f"Lattice parameters accepted: {self.cell}")

        except ValueError as e:
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Please enter valid numerical values for all parameters.\n\nError: {str(e)}"
            )

    def get_cell(self) -> Optional[Cell]:
        """Return the cell object if parameters were accepted."""
        return self.cell
