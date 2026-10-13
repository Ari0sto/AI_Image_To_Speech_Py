# AI Vision & Voice Assistant

A Python application that analyzes images using Hugging Face AI models and generates spoken descriptions.

The project combines **Image Captioning** and **Text-to-Speech (TTS)** in a single workflow. It supports multiple images, text description storage, WAV audio generation, and speech regeneration without repeating image analysis.

## Features

- Load JPG, JPEG, and PNG images.
- Generate English image descriptions using BLIP.
- Convert generated descriptions into speech using MMS-TTS.
- Save audio files in WAV format.
- Automatically generate unique audio filenames.
- Save descriptions in `output/descriptions.txt`.
- Process multiple images during one session.
- Regenerate speech without analyzing the image again.
- Handle invalid images and model processing errors.
- Support NVIDIA CUDA acceleration with CPU fallback.
- Navigate the application through a console menu.

## Technologies

- Python
- Hugging Face Transformers
- PyTorch
- Pillow
- SciPy
- Hugging Face Hub

## AI Models

### 1. Image Captioning — BLIP

**Model:** [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

BLIP (Bootstrapping Language-Image Pre-training) is a vision-language model that combines visual feature extraction with natural language generation.

The model processes an image and generates an English caption describing its main visual content. Its architecture uses a Vision Transformer (ViT) backbone for image processing.

**Limitations:** The model may overlook objects, misidentify visual details, or produce inaccurate descriptions.

### 2. Text-to-Speech — MMS-TTS

**Model:** [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)

MMS-TTS is an English text-to-speech model developed by Meta. It uses the VITS architecture to generate speech waveforms from input text.

The generated audio is saved as a WAV file using the sampling rate returned by the model.

**Limitations:** Speech pronunciation, pauses, and intonation may not always sound natural.

**License:** CC BY-NC 4.0 (non-commercial use).

## Project Structure

    AI_Image_To_Speech_Py/
    ├── input/
    │   └── ... test images
    ├── output/
    │   ├── descriptions.txt
    │   └── ... generated WAV files
    ├── main.py
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Installation

### 1. Clone the repository

    git clone <repository-url>
    cd AI_Image_To_Speech_Py

Replace `<repository-url>` with the actual GitHub repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

### 3. Install dependencies

    python -m pip install -r requirements.txt

For NVIDIA GPU acceleration, install a compatible CUDA-enabled PyTorch build following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Without a supported GPU, the application can run on CPU.

### 4. Configure the model cache

By default, Hugging Face downloads model files to the user's cache directory.

This project can be configured to use a custom cache location, for example:

    D:/HuggingFace_Cache

Change the `HF_HOME` environment setting in `main.py` if necessary.

## Usage

Run the application:

    python main.py

The console menu provides three options:

1. **Analyze a new image** — enter an image path, generate a description, and save a WAV file.
2. **Regenerate speech** — generate another audio file from the latest description without repeating image analysis.
3. **Exit** — close the application.

Example image path:

    input/cat.jpg

Example generated description:

    a pink cat with a mustache on its head

Example output:

    output/cat.wav
    output/descriptions.txt

If an audio file with the same name already exists, a numeric suffix is added to prevent overwriting.

Models are loaded once when the program starts and reused during the session. On the first run, model weights are downloaded from Hugging Face.

## Processing Workflow

1. The user selects an image.
2. Pillow opens the image and converts it to RGB.
3. BLIP generates an English text description.
4. MMS-TTS synthesizes speech from the description.
5. The application saves the audio as a WAV file.
6. The text description is appended to `descriptions.txt`.
7. The output paths and model names are displayed in the console.

## Testing and Results

The application was tested with five images.

The evaluation focuses on:

- Correct recognition of the main objects.
- Accuracy of the generated descriptions.
- Missing or incorrectly identified image details.
- Speech intelligibility and pronunciation.
- Successful creation of WAV files.

Detailed results will be added after reviewing the actual images and their generated descriptions.

## Known Limitations

- Descriptions are generated in English only.
- Image captions may contain incorrect visual information.
- Repeated analysis of the same image does not reuse its previous caption automatically.
- The most recent description is stored in memory and is not restored after restarting the program.
- Generated speech quality depends on the TTS model.
- Model files require additional disk space.
- The application currently uses one TTS model.

## References

- [BLIP — Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [MMS-TTS English — Hugging Face](https://huggingface.co/facebook/mms-tts-eng)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch](https://pytorch.org/)
- [BLIP Research Paper](https://arxiv.org/abs/2201.12086)
- [MMS Research Paper](https://arxiv.org/abs/2305.13516)

## Project Status

The core application and its main additional features are implemented.