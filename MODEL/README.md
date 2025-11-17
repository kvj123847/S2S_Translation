S2S_Translation

This repository contains a small sequence-to-sequence (S2S) translation project, originally imported from a local workspace.

Project Contents

interface.html — simple HTML interface for the translator

translator.html — additional UI layout for translation

main2.py — main Python script to run the translator

model2.py — model definitions and helper functions



Fine-Tuned Model

The translation model used in this project is hosted on Hugging Face:

Model link: https://huggingface.co/kklwq/whisper-small-hi-finetuned



Quick Start

Ensure Python 3.8 or later is installed.

From the project root, run:

python main2.py

Notes

A .gitignore is recommended to avoid committing build artifacts and cache files.

Suggested .gitignore:

__pycache__/
*.py[cod]


You can extend this with virtual environment folders or other project-specific exclusions if needed.
