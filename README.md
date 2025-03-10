🩺 Diabetic Retinopathy Classification - EfficientNet-B3
📌 Overview
This project was developed for the AI/ML Hackathon 2025, focusing on Diabetic Retinopathy (DR) detection using deep learning. We implemented an EfficientNet-B3 model to classify retinal images into five severity levels of diabetic retinopathy.

📂 Project Structure

Diabetic_Retinopathy_Classification/
├── model/
│   └── efficientnet_b3_retinopathy.pt  # Trained model weights (Optional, if available)
├── notebooks/
│   └── model_training.ipynb            # Kaggle notebook (training & evaluation)
├── report.pdf                          # Project documentation (methodology & results)
├── README.md                           # Project overview & setup guide
├── requirements.txt                    # Dependencies used in the project
🖼️ Dataset
The dataset consists of retinal fundus images, classified into five categories based on Diabetic Retinopathy (DR) severity:

Label	Category	Description
0	No DR	Healthy retina
1	Mild DR	Early-stage diabetic retinopathy
2	Moderate DR	Moderate damage observed
3	Severe DR	Significant damage detected
4	Proliferative DR	Most severe condition
Dataset Source:
Kaggle: Diabetic Retinopathy Balanced Dataset

⚙️ Model Architecture & Training
Architecture: EfficientNet-B3 (Pretrained on ImageNet)
Input Size: 224x224 pixels (RGB images)
Loss Function: CrossEntropyLoss
Optimizer: Adam (lr=0.0001)
Batch Size: 16
Epochs: 10-20 (Early stopping applied)
Learning Rate Scheduler: ReduceLROnPlateau (Patience: 3)
Augmentation: Random horizontal flip, rotation (±15°), brightness & contrast adjustments
📊 Model Performance (Test Data)
Our model achieved 86% accuracy on previously unseen test data.

Class	Precision	Recall	F1-Score	Support
No DR	0.75	0.69	0.72	1000
Mild DR	0.80	0.87	0.84	971
Moderate DR	0.78	0.77	0.77	1000
Severe DR	0.98	0.99	0.99	1000
Proliferative DR	0.99	1.00	0.99	1000
✅ Final Accuracy: 86%
✅ High Precision for Severe and Proliferative DR Cases

🔎 Explainability - Grad-CAM Visualizations
To improve model interpretability, we applied Grad-CAM visualization to highlight the important retinal areas influencing the model’s decision.

Key Findings:

The model focused on central retinal regions, which aligns with clinical diagnostic methods.
High-risk cases (Severe & Proliferative DR) were clearly detected.
This enhances trust and reliability in medical AI applications.
🚀 How to Run this Project
1️⃣ Clone the Repository

git clone https://github.com/yourusername/Diabetic_Retinopathy_Classification.git
cd Diabetic_Retinopathy_Classification
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Run Jupyter Notebook

jupyter notebook notebooks/model_training.ipynb
4️⃣ Load the Trained Model (Optional, if model weights provided)
python

import torch
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
model.load_state_dict(torch.load('model/efficientnet_b3_retinopathy.pt', map_location=device))
model = model.to(device)
model.eval()
📝 Dependencies
All required dependencies are listed in requirements.txt. Install them using:

pip install -r requirements.txt
Contents of requirements.txt:

torch
torchvision
efficientnet_pytorch
matplotlib
scikit-learn
tqdm
pytorch-grad-cam
📜 Future Work & Improvements
🔹 Upgrade Model: Switch to EfficientNet-B5 or use an ensemble approach for higher accuracy.
🔹 Class-Weighted Loss: Improve detection of early-stage DR cases (Mild/Moderate).
🔹 Deployment: Convert to FastAPI or TensorRT for faster real-time prediction.
