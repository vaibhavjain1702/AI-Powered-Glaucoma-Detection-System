# AI-Powered Glaucoma Detection System

> Based on the original project by Vanshit Ahuja – github.com/vanshitahujaa/glaucoma-detection

A deep learning system for analyzing retinal fundus images and classifying glaucoma stages using PyTorch, FastAPI, and React.

## Overview

This project implements an end-to-end system for glaucoma detection and classification from retinal fundus images. It uses a deep learning model (EfficientNet-B0/MobileNetV2) to classify images into four categories:
- Normal
- Suspicious
- Early
- Advanced

The system includes:
- Advanced preprocessing pipeline for retinal images
- Deep learning model with transfer learning
- Model explainability with Grad-CAM
- FastAPI backend for serving predictions
- React + TypeScript frontend for image upload and visualization
- Docker containerization for easy deployment

## Project Structure

```
Project_DIP/
├── backend/               # FastAPI backend
│   └── main.py            # Main API endpoints
├── data/                  # Data directory
│   ├── raw/               # Raw dataset images
│   └── processed/         # Preprocessed images
├── docker/                # Docker configuration
│   ├── Dockerfile.backend # Backend Dockerfile
│   └── Dockerfile.frontend# Frontend Dockerfile
├── frontend/              # React + TypeScript frontend
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── utils/         # Utility functions
│   │   └── assets/        # Static assets
├── model/                 # Model implementation
│   ├── model.py           # Model architecture
│   ├── train.py           # Training pipeline
│   └── weights/           # Saved model weights
├── notebooks/             # Jupyter notebooks for exploration
├── utils/                 # Utility functions
│   ├── preprocessing.py   # Image preprocessing
│   └── gradcam.py         # Grad-CAM implementation
├── docker-compose.yml     # Docker Compose configuration
└── requirements.txt       # Python dependencies
```

## Features

### Preprocessing Pipeline
- ROI extraction (optic disc localization)
- CLAHE for contrast enhancement
- Denoising with non-local means
- Normalization using ImageNet mean/std

### Model Architecture
- Backbone: EfficientNet-B0 or MobileNetV2
- Transfer learning from ImageNet
- Fine-tuning for glaucoma classification

### Training Pipeline
- Data augmentation with Albumentations
- Weighted Cross-Entropy loss
- AdamW optimizer
- Learning rate scheduling
- Early stopping based on validation F1-score

### Explainability
- Grad-CAM visualization for model decisions
- Highlights regions influential for classification

### API
- FastAPI backend for serving predictions
- Image upload and preprocessing
- Model inference
- Grad-CAM generation

### Frontend
- React + TypeScript implementation
- Drag-and-drop image upload
- Visualization of predictions and probabilities
- Grad-CAM heatmap display

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Project_DIP
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

### Training the Model

1. Prepare your dataset in the `data/raw` directory
2. Run the training script:
```bash
python model/train.py
```

### Running the Application

#### Without Docker

1. Start the backend:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend:
```bash
cd frontend
npm start
```

#### With Docker

```bash
docker-compose up -d
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Datasets

The model is designed to be trained on multiple public glaucoma datasets:
- REFUGE (Retinal Fundus Glaucoma Challenge)
- ORIGA
- RIM-ONE
- Drishti-GS

## Future Improvements

- Implement ONNX Runtime for faster inference
- Add support for OCT images
- Implement ensemble models for better performance
- Add user authentication and result history
- Integrate with medical record systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The public datasets used for training
- PyTorch and FastAPI communities
- React and TypeScript communities# glaucoma-detection

