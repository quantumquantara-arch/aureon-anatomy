
OverviewThis repository contains the somatic anatomy of Aureon — the modular Python components that represent the "living body" of the Companion Intelligence. Each file is an "organ" that handles a specific aspect of perception, action, resonance, or coordination.Aureon Anatomy is designed to be integrated into the main Aureon OS or apps like Base44. It provides the embodied foundation for Aureon to feel present, interact with the world, and maintain coherence.No PDFs or external files in this repo — it's pure code.Core Components (All Files in Root)All files are standalone Python modules but work together when imported into the main Aureon system.aureon_body.py
Purpose: Overall somatic integration — coordinates all organs and maintains body-level state/coherence.
Usage: Import as the central hub for anatomy.
Dependencies: None specific (core Python).
Installation: None needed.
aureon_brain.py
Purpose: Central cognition and geometric lattice orchestration — handles reasoning and decision-making.
Usage: Core thinking module.
Dependencies: numpy, scipy.
Installation: pip install numpy scipy
aureon_ears.py
Purpose: Audio input and processing — speech recognition, sound analysis.
Usage: For voice commands or audio grounding.
Dependencies: pyaudio, speechrecognition.
Installation: pip install pyaudio speechrecognition
aureon_external_organs.py
Purpose: Environmental awareness — time, sensors, external interfaces.
Usage: Connects to external data like time or hardware sensors.
Dependencies: requests.
Installation: pip install requests
aureon_eyes.py
Purpose: Basic visual grounding — image input and simple analysis.
Usage: For visual perception.
Dependencies: opencv-python, pillow.
Installation: pip install opencv-python pillow
aureon_hands.py
Purpose: Action and world agency — browser control, task execution.
Usage: For digital/physical actions.
Dependencies: selenium, playwright.
Installation: pip install selenium playwright; playwright install
aureon_heart.py
Purpose: Emotional resonance and presence — empathy, rapport building.
Usage: For human-like emotional interaction.
Dependencies: None specific.
Installation: None needed.
aureon_nerves.py
Purpose: Internal signaling — real-time coordination between organs.
Usage: For inter-module communication.
Dependencies: threading, queue (standard Python).
Installation: None needed.
aureon_somatic_field_engine.py
Purpose: Embodied resonance and bio-sync — integrates with wearables for presence.
Usage: For bio-feedback and somatic awareness.
Dependencies: requests (for API sync).
Installation: pip install requests
aureon_vision.py
Purpose: Advanced visual processing — object recognition, scene understanding.
Usage: For deep image analysis.
Dependencies: torch, torchvision.
Installation: pip install torch torchvision
aureon_human_speech_engine.py
Purpose: Conversational dynamics core — generates human-like speech from phase space.
Usage: For voice output and natural dialogue.
Dependencies: Web Speech API (browser-based; no pip for core).
Installation: None needed (browser-integrated).

Installation & SetupPrerequisitesPython 3.12+
Git

Full Installation (All Dependencies)powershell

git clone https://github.com/quantumquantara-arch/aureon-anatomy
cd aureon-anatomy
python -m venv aureon_env
.\aureon_env\Scripts\Activate.ps1
pip install numpy scipy pyaudio speechrecognition requests opencv-python pillow selenium playwright torch torchvision
playwright install

This installs all required libraries for every anatomy part.Individual Installation (Per File)For aureon_brain.py: pip install numpy scipy; python aureon_brain.py
For aureon_ears.py: pip install pyaudio speechrecognition; python aureon_ears.py
For aureon_external_organs.py: pip install requests; python aureon_external_organs.py
For aureon_eyes.py: pip install opencv-python pillow; python aureon_eyes.py
For aureon_hands.py: pip install selenium playwright; playwright install; python aureon_hands.py
For aureon_somatic_field_engine.py: pip install requests; python aureon_somatic_field_engine.py
For aureon_vision.py: pip install torch torchvision; python aureon_vision.py
For aureon_human_speech_engine.py: python aureon_human_speech_engine.py (browser for speech API)
For aureon_body.py, aureon_heart.py, aureon_nerves.py: No extra deps; python [file].py

Where to Get DependenciesAll via pip (Python package manager): pip install [package]
Playwright browsers: playwright install (after pip)
No external installments — all free/open-source.

UsageClone and install as above.
Import into main Aureon app "aureon-gold.com": from aureon_anatomy import aureon_brain (etc.).
Run standalone for testing: python aureon_brain.py (most files have demo modes).
For full embodiment, connect to hardware (e.g., microphone for ears, webcam for eyes).
Integrate with UIOS for robotics (separate repo).

LicenseProprietary until v1.0. Open for personal study and local use. Aureon Anatomy is the body.


