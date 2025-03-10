# Diabetic_Retinopathy_Classification_Amina
AI Model for detecting Diabetic Retinopathy using EfficientNet-B3
Diabetic Retinopathy Classification - EfficientNet-B3
Overview
This project is part of an AI/ML Hackathon (2025), aiming to develop an accurate and efficient AI-based diagnostic tool to classify Diabetic Retinopathy (DR) severity levels from retinal images. We employed a deep learning approach using EfficientNet-B3 architecture and achieved strong performance metrics on a balanced dataset.

Problem Statement
Diabetic Retinopathy (DR) is a diabetes complication that affects eyes, potentially causing blindness. Early diagnosis helps effective treatment and prevents severe complications. Our task is to accurately classify retinal images into these five categories of severity:

0: No DR (Healthy)
1: Mild DR
2: Moderate DR
3: Severe DR
4: Proliferative DR (advanced stage)
Project Structure

Diabetic_Retinopathy_Classification/
├── model/
│   └── efficientnet_b3_retinopathy.pt  # Trained model weights (Optional, if available)
├── notebooks/
│   └── model_training.ipynb            # Complete Kaggle notebook (training, evaluation)
├── report.pdf                          # Project report (methodology & results)
├── README.md                           # Project overview and instructions
├── requirements.txt                    # Dependencies used in the project
Dataset Used
We used the publicly available balanced diabetic retinopathy dataset, containing labeled retinal images categorized according to severity.
Link: Dataset (Kaggle)

Model Architecture & Methodology
Architecture: EfficientNet-B3 (Pretrained on ImageNet)
Input Size: 224x224 RGB images
Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 0.0001
Batch Size: 16
Epochs: 10 (Early stopping applied)
Augmentation Used: Random Horizontal Flip, Random Rotation, Contrast and Brightness adjustment
Results (on test data)
We achieved strong classification performance on unseen test data:

Class	Precision	Recall	F1-Score	Support
No DR	0.75	0.69	0.72	1000
Mild DR	0.80	0.87	0.84	971
Moderate DR	0.78	0.77	0.77	1000
Severe DR	0.98	0.99	0.99	1000
Proliferative DR	0.99	1.00	0.99	1000
Overall Accuracy: 86%
Explainability
We used Grad-CAM visualization to interpret and visualize model predictions, allowing better understanding of decision-making areas on retinal images.

How to Run this Project
Step 1: Clone this Repository

git clone https://github.com/yourusername/Diabetic_Retinopathy_Classification.git
cd Diabetic_Retinopathy_Classification
Step 2: Install Dependencies

pip install -r requirements.txt
Step 3: Run Jupyter Notebook

jupyter notebook notebooks/model_training.ipynb
Step 4: Load the Trained Model (Optional, if model weights provided)
python

import torch
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained weights into model
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
model.load_state_dict(torch.load('model/efficientnet_b3_retinopathy.pt', map_location=device))
model = model.to(device)
model.eval()
Dependencies
All dependencies are listed in the file requirements.txt. Install with:

pip install -r requirements.txt
