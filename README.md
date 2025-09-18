
# 🦾 Model-Based Digital Twin: Functional Mock-Up Units (FMUs) for Industrial Robotics

This repository hosts modular, FMI 2.0-compliant Functional Mock-Up Units (FMUs) developed for Digital Twin (DT) applications in industrial robotic systems. Focused primarily on the UR10e collaborative robot, the FMUs here offer validated forward kinematics modeling based on real-world robot data and enable portable, platform-independent simulation capabilities.

## 🔍 Project Overview

Digital twins are revolutionizing how robotic systems are simulated, monitored, and deployed. This project presents the creation and validation of FMU-based kinematic models for the UR10e Cobot, compliant with the **FMI 2.0 Co-Simulation** standard (see: https://github.com/modelica/fmi-standard). 

The FMUs are developed using open-source Python toolchains such as **UniFMU** and tested in environments like **FMPy GUI** for robust cross-platform simulation.

## 📌 Featured Deliverables

### 📘 [Technical Report – UR10e FK FMU Model Validation](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/blob/main/docs/Technical%20Report%20UR10e%20Cobot%20Arm%20Forward%20Kinematics%20Functional%20Mock-Up%20Unit%20Model%20Validation.pdf)

A complete technical documentation of the development, implementation, and validation of the **[UR10e](https://www.universal-robots.com/tr/urunler/ur10-robot/) Forward Kinematics FMU**, including:

- Denavit–Hartenberg-based kinematic modeling
- FMU generation and internal architecture
- Simulation via real robot joint data
- Statistical comparison (MAE, RMSE) with physical measurements
- Orientation tracking with quaternion normalization

👉 Ideal for researchers and engineers looking to understand the FMU pipeline in industrial robotics.

---

### 📘 [Technical Report – UR10e IK FMU Model Validation](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/blob/main/docs/Technical%20Report%20UR10e%20Cobot%20Arm%20Inverse%20Kinematics%20Functional%20Mock-Up%20Unit%20Model%20Validation.pdf)

A complete technical documentation of the development, implementation, and validation of the **[UR10e](https://www.universal-robots.com/tr/urunler/ur10-robot/) Inverse Kinematics FMU**, including:

- IKPy based FMU modeling
- FMU generation and internal architecture
- Simulation via real robot's end-effector and orientation data from pendant
- Statistical comparison (MAE, RMSE) with physical measurements
- Orientation tracking with quaternion normalization

👉 Ideal for engineers looking to understand the kinematics solution in FMU pipeline in digital twins.

---

### 🧾 [FMU Types Catalogue (v0.3)](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/blob/main/docs/FMU_Types_Catalogue/RobotManipulatorDynamics_v0.3.pdf)

An indexed overview of all FMUs developed in this project so far, including:

- **[UR10e Forward Kinematics](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/tree/main/Cobot_Ur10e/UR10e_FK/FMU_Ur10e_FK_Source_Codes)**

- **[UR10e Inverse Kinematics](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/tree/main/Cobot_Ur10e/UR10e_IK)**
- Dynamics and HIL-ready modules (upcoming)

The catalogue is versioned and updated to reflect the current state of modular FMU libraries developed for robotic manipulators.

---

### 🧠 [FMU Raw Source Code – UR10e FK](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/tree/main/Cobot_Ur10e/UR10e_FK/Ur10e_FK_Source_Codes)

Core implementation in Python for:

- Homogeneous transformation logic
- Quaternion and Euler converters
- FMU interface (FMI 2.0-compliant wrapper)
- Self-test modules for standalone simulation

All the code is cleanly modularized in `model.py` and supports both FMU packaging and direct CLI-based testing.

### 🧠 [FMU Raw Source Code – UR10e IK](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/tree/main/Cobot_Ur10e/UR10e_IK/UR10e_IK_files)

Core implementation in Python for:

- Inverse Kinematics Engine :Built upon the ikpy library, leveraging Denavit–Hartenberg parameters of the UR10e. Includes Jacobian-based numerical solving and deterministic elbow-down branch selection.

- Canonicalization Logic: Implements post-processing of raw IK solutions (e.g., canonicalize_q2_q4) to consistently enforce UR10e’s elbow-down convention and prevent oscillations between equivalent solution branches.

- FMU Interface (FMI 2.0-compliant): Encapsulated in Model class (model.py) which maps FMU inputs (pose vectors) to outputs (joint angles). Implements all FMI callbacks (set_real, get_real, do_step, serialize/deserialize) for co-simulation.

- Transformation Utilities: Functions for pose → homogeneous transform construction (build_transform), angle wrapping (wrap_to_pi), and FK validation checks against DH parameters.

- Seed Management & Stability: Maintains last valid joint configuration across solver calls to improve convergence stability in iterative IK solving.

---

### 📊 [Comparison Report – UR10e FK vs. Real Robot](https://github.com/ESOGU-SRLAB/Model-Based-Digital-Twin-Functional-Mockup-Unit/blob/main/Cobot_Ur10e/UR10e_FK/RealUR10e-FMU_output_analysis.pdf)

A quantitative and visual analysis comparing the FMU outputs with joint telemetry from a real UR10e cobot. Includes:

- Position errors over time (mm-level accuracy)
- Quaternion distance metrics
- RMSE/MAE summary tables
- Overlay plots of simulated vs real trajectories


---

## 🔧 Toolchain and Dependencies

- **Python 3.8+**
- [UniFMU](https://github.com/INTO-CPS-Association/unifmu)
- [FMPy](https://github.com/CATIA-Systems/FMPy)
- `numpy`, `scipy`, `pandas`, `matplotlib`

> 💡 FMUs are exported using `unifmu generate python UR10e_FK` and are FMI 2.0 Co-Simulation compatible.

---

## 🚀 Use Cases

- Real-time Digital Twin monitoring systems
- Simulation-based commissioning and predictive planning
- Embedded HIL platforms using robot state feedback
- Modular robotics training platforms

---

## 📁 Repository Structure

```bash
Model-Based-Digital-Twin-Functional-Mockup-Unit/
│
├── docs/                        # Technical reports and presentations
│   ├── Technical Report ...docx
│   └── FMU_Types_Catalogue/
│       └── RobotManipulatorDynamics_v0.3.pptx
│
├── Cobot_Ur10e/
│   └── UR10e_FK/
│       ├── FMU_Ur10e_FK_Source_Codes/
│       ├── UR10e_FK.fmu
│       ├── fmpy_inputs.csv
│       ├── ur10e_outs.xlsx
│       └── RealUR10e-FMU_output_analysis.pdf
```

---

## 👤 Authors and Contact

**[Serhat Kahraman](https://github.com/Serhatkahraman1)**, **[ESOGU ASRLab](https://github.com/ESOGU-SRLAB)**  
📧 For academic inquiries, integration questions, or collaborations, please contact via GitHub issues or your institutional email.

---

Stay tuned for further improvements!

## 📜 License

MIT License.  
Feel free to use, modify, and extend with citation.
