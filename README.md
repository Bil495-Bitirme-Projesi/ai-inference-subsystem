# 🎥 Video Anomaly Detection (VAD) Subsystem

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Tool](https://img.shields.io/badge/build-uv-blueviolet.svg)](https://github.com/astral-sh/uv)

This repository contains the AI inference subsystem for the **Video Anomaly Detection** project. It utilizes a fine-tuned **VideoMAE** architecture to detect and classify anomalous events (e.g., Fighting, Robbery, Burglary) in surveillance video streams.

---

## 🚀 Getting Started

### 📋 Prerequisites

- **Python 3.12** or higher.
- **uv** (Highly recommended for lightning-fast dependency management).
  - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 🛠 Installation & Build

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-inference-subsystem
   ```

2. **Sync dependencies:**
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   This will automatically create a virtual environment and install all necessary packages including `torch`, `transformers`, and `opencv-python`.

---

## 🧠 Model Setup (CRITICAL)

The project requires the **VideoMAE-Large** model fine-tuned on the **UCF-Crime** dataset. Due to file size limitations, the model weights are not included in the repository and must be downloaded manually from Hugging Face.

1. **Download the model:**
   Visit [OPear/videomae-large-finetuned-UCF-Crime](https://huggingface.co/OPear/videomae-large-finetuned-UCF-Crime) on Hugging Face.

2. **Prepare the directory:**
   ```bash
   mkdir -p models/OPear
   ```

3. **Install the files:**
   Download and place the following files into `models/OPear/`:
   - `config.json`
   - `model.safetensors` (or `pytorch_model.bin`)
   - `preprocessor_config.json` (if available)

   **Your directory structure should look like this:**
   ```text
   vad-python/
   └── models/
       └── OPear/
           ├── config.json
           └── model.safetensors
   ```

---

## 🏃 Usage

### Running Inference
Currently, the main entry point is `main.py`. Ensure your configuration in `config/videomae_cfg.json` points to the correct model path.

```bash
uv run main.py
```

### Running Tests
The project includes a suite of tests to verify the inference engine and data ingestion:

```bash
uv run python -m unittest discover tests
```

---

## 📂 Project Structure

| Directory | Description |
| :--- | :--- |
| `source/` | Core logic, including inference engines and dispatchers. |
| `models/` | Local storage for pre-trained model weights. |
| `config/` | System and model-specific configuration files (JSON). |
| `tests/` | Unit and integration tests. |
| `sample_data/` | Sample videos for testing purposes. |

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---
*Developed as part of the BIL495 Bitirme Projesi.*
