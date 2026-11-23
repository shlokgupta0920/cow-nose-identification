# 🐄 Cow Nose Identification System

An AI-based educational prototype for cow registration and nose-image similarity comparison.

## About

The Cow Nose Identification System is a Streamlit web application that lets users register cow details with nose images and compare a new image against the registered images.

It uses DINOv2 (`facebook/dinov2-small`) to extract image features and cosine similarity to rank visually similar registered images.

> **Important:** This is an experimental demonstration. DINOv2 is a general-purpose image model, not a model specifically trained or validated for cattle nose biometrics. Similarity scores are not proof of identity and should not be used for real-world animal identification decisions.

## Features

- Register cow details (ID, name, age, breed and owner information)
- Save multiple nose images for each registered cow
- Compare an uploaded or camera-captured image with registered images
- Display the closest image match and the associated cow record
- View registered cow records through a Streamlit interface
- Validate contact numbers using country selection

## Tech Stack

- Python
- Streamlit
- PyTorch and Torchvision
- Hugging Face Transformers
- DINOv2
- Pillow
- NumPy
- phonenumbers

## Project Files

```text
cow-nose-identification/
├── app.py
├── ai_recognition.py
├── requirements.txt
├── cows.json          # created/used locally by the app
└── cow_images/        # image storage used by the app
```

The `cows.json` file and `cow_images/` directory may be created locally when the app is used. Do not commit private or real owner contact information or animal records to a public repository.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shlokgupta0920/cow-nose-identification.git
cd cow-nose-identification
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

The first run may take longer because the DINOv2 model needs to be downloaded.

## AI Recognition Workflow

1. The app receives a query nose image.
2. DINOv2 extracts a feature representation for the query image.
3. The app extracts features from registered nose images.
4. Cosine similarity is used to rank the image comparisons.
5. The app displays the closest match according to its configured threshold.

The threshold is a prototype setting and has not been scientifically calibrated. Performance can vary with image quality, lighting, camera angle and the images available in the records.

## Live Demo

[Open the Streamlit app](https://cow-nose-identification-by-shlok.streamlit.app)

## Developer

**Shlok Gupta**

- GitHub: [@shlokgupta0920](https://github.com/shlokgupta0920)
- Repository: [cow-nose-identification](https://github.com/shlokgupta0920/cow-nose-identification)

## License

No open-source license has been specified for this repository yet. Unless a license is added, the default copyright rules apply; please contact the repository owner before reusing or distributing this project.
