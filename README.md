# 🚧 Predictive Road Maintenance — Streamlit Demo

This project is a **demo web app** for predictive road maintenance.  
It takes a **road image**, predicts its **condition** using a CNN, then estimates **months until deterioration** and a **risk score (0–100)** using traffic & weather metadata.

- I developed a Predictive Road Maintenance demo to showcase how AI can support infrastructure planning. The situation was that manual road inspections are reactive and costly, and there was a need for a proactive solution.
- My task was to build an app that could assess road condition from images and forecast when deterioration would occur. For the action, I used a MobileNetV2-based CNN to classify road images into four categories (Good, Satisfactory, Poor, Very Poor) and a scikit-learn regression pipeline to predict the number of months until a segment becomes “Poor,” integrating metadata such as traffic, rainfall, pavement type, and maintenance history.
- I also designed a risk score formula, normalized it to 0–100, and implemented the whole workflow in a Streamlit web app with a clean UI, model caching, and fallback logic. The result was a working end-to-end system achieving ~85% image classification accuracy and ~0.70 R² on the regression model, enabling actionable insights for prioritizing road repairs and serving as a strong portfolio project to demonstrate skills in Computer Vision, ML pipelines, and full-stack AI app deployment.
---

## ✨ Features

- 🖼️ **Image Classification** — MobileNetV2 predicts condition:
  *Good / Satisfactory / Poor / Very Poor*  
- 📊 **Tabular ML Model** — scikit-learn regression pipeline predicts **months_to_poor**.  
- ⚠️ **Risk Score** — Combines months prediction with traffic & heavy truck share to compute urgency.  
- 🖥️ **Streamlit UI** — Simple, interactive web app.  
- 🔄 **Fallback Logic** — Nearest-neighbor search in dataset if predictions are unrealistic.  

---

## 📂 Project Structure
```
├── app.py # Streamlit app
├── requirements.txt # Python dependencies
├── models/
│ ├── road_condition_mobilenetv2_aug.h5
│ └── months_to_poor_model.joblib
├── data/
│ └── simulated_road_deterioration.csv
└── docs/ (optional screenshots)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/road-maintenance-demo.git
cd road-maintenance-demo
# Create environment & install dependencies
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt

#Run Streamlit
streamlit run app.py
```
## Models
- ### Road Condition Classifier
- - Architecture: MobileNetV2 (TensorFlow/Keras)
  - Input: 224×224 RGB image
  - Output: One of 4 classes (Good, Satisfactory, Poor, Very Poor)

- ### Months-to-Poor Regressor
- - Features: traffic/day, % heavy trucks, rainfall, pavement type, months since last maintenance, condition_now
  - Output: Float months until the road becomes “Poor” or worse
  - Pipeline: ColumnTransformer (scale + one-hot) → RandomForestRegressor (example)

 ## Example Workflow
- Upload a road image.
- Classifier predicts: e.g., “Very Poor” (confidence 0.65)
- Enter metadata: traffic volume, rainfall, trucks %, pavement type, etc.
- Click Predict → outputs:
- Months until deterioration (integer)
- Risk score (0–100)
- Message with urgency level

## License
- MIT License (or your choice)

## Acknowledgements
- MobileNetV2 (Sandler et al.)
- Streamlit for rapid app deployment
- scikit-learn for regression pipelines
