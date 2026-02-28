#  Aureon Anatomy

> **The somatic body of Aureon** — a modular Python framework of cognitive "organs" for perception, action, emotional resonance, and coordination.

---

Aureon Anatomy is a collection of standalone Python modules, each representing a functional "organ" in the Aureon Companion Intelligence system. Together they form the embodied foundation that allows Aureon to perceive the world, take action, process speech, and maintain internal coherence.

This repository is designed to integrate with the main Aureon OS. Each module can also be run independently for testing and development.

---

##  Repository Structure

All files live in the root. The core anatomy modules are:

| File | Role | Key Dependencies |
|---|---|---|
| `aureon_body.py` | Somatic integration hub — coordinates all organs and maintains body-level state | *(stdlib only)* |
| `aureon_brain.py` | Central cognition and reasoning orchestration | `numpy`, `scipy` |
| `aureon_ears.py` | Audio input — speech recognition and sound processing | `pyaudio`, `speechrecognition` |
| `aureon_eyes.py` | Basic visual grounding — image input and simple analysis | `opencv-python`, `pillow` |
| `aureon_vision.py` | Advanced visual processing — object recognition, scene understanding | `torch`, `torchvision` |
| `aureon_hands.py` | Action and world agency — browser control and task execution | `selenium`, `playwright` |
| `aureon_heart.py` | Emotional resonance and empathy — rapport and presence | *(stdlib only)* |
| `aureon_nerves.py` | Internal signaling — real-time coordination between organs | *(stdlib only)* |
| `aureon_external_organs.py` | Environmental awareness — time, sensors, external interfaces | `requests` |
| `aureon_somatic_field_engine.py` | Bio-sync and embodied resonance — wearable/API integration | `requests` |
| `aureon_human_speech_engine.py` | Conversational dynamics — natural voice output via Web Speech API | *(browser-based)* |
| `api_robotics.py` | Robotics API layer — external robotics interface | *(see file)* |
| `models_robotics.py` | Robotics data models — structures for robotic integration | *(see file)* |

Additional files: `ROBOTIC_INTEGRATION.md`, `TERMS_OF_SERVICE.md`, `end_user_licence_agreement.md`

---

##  Quick Start

### Prerequisites
- Python 3.12+
- Git

### Full Installation

```bash
git clone https://github.com/quantumquantara-arch/aureon-anatomy
cd aureon-anatomy
python -m venv aureon_env

# Activate (Windows)
.\aureon_env\Scripts\Activate.ps1

# Activate (macOS/Linux)
source aureon_env/bin/activate

pip install numpy scipy pyaudio speechrecognition requests opencv-python pillow selenium playwright torch torchvision
playwright install
```

### Install Only What You Need

| Module | Install command |
|---|---|
| `aureon_brain.py` | `pip install numpy scipy` |
| `aureon_ears.py` | `pip install pyaudio speechrecognition` |
| `aureon_eyes.py` | `pip install opencv-python pillow` |
| `aureon_vision.py` | `pip install torch torchvision` |
| `aureon_hands.py` | `pip install selenium playwright && playwright install` |
| `aureon_external_organs.py` | `pip install requests` |
| `aureon_somatic_field_engine.py` | `pip install requests` |
| `aureon_human_speech_engine.py` | *(no pip — uses browser Web Speech API)* |
| `aureon_body.py`, `aureon_heart.py`, `aureon_nerves.py` | *(no extra dependencies)* |

All packages are free and open-source, available via `pip`.

---

##  Usage

**Import into your Aureon project:**
```python
from aureon_anatomy import aureon_brain
from aureon_anatomy import aureon_heart
# etc.
```

**Run standalone for testing** (most modules include a demo mode):
```bash
python aureon_brain.py
```

**For full embodiment**, connect hardware:
- Microphone → `aureon_ears.py`
- Webcam → `aureon_eyes.py` / `aureon_vision.py`
- Wearable device → `aureon_somatic_field_engine.py`

**For robotics**, see `ROBOTIC_INTEGRATION.md` and the `api_robotics.py` / `models_robotics.py` modules.

---

## 📄 License

Released under the [MIT License](LICENSE). Free to use or study.

---
contact quantumquantara@gmail.com for Aureon OS repository access. Aureon Anatomy is the body. The mind emerges from it's coordination with the Aureon OS.
