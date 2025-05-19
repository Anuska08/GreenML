# 🌱⚡ GreenML Dashboard – Smart Appliance Energy Predictor

The **GreenML Dashboard** is a sustainable AI initiative that leverages **machine learning** and **green computing** to accurately predict household appliance energy consumption based on indoor and outdoor environmental conditions such as temperature, humidity, occupancy, and CO₂ levels.

> What sets this project apart is its integration with the `codecarbon` library to track the **carbon emissions** produced during model training, promoting **eco-conscious AI practices**.

## 🔧 Tech Stack

- **Python** – Data handling and model development  
- **scikit-learn** – ML algorithms (Linear Regression, Random Forest)  
- **codecarbon** – Carbon footprint measurement  
- **seaborn** & **matplotlib** – Visual analytics  
- **Streamlit** – Interactive web dashboard  

## 📌 Key Features

✅ Model performance comparison (MAE, R², CO₂ emissions)  
✅ Heatmap showing top 10 feature correlations  
✅ Single input prediction and batch CSV-based predictions  
✅ Energy usage and environmental impact visualizations  
✅ Streamlit-powered UI – **ready for public deployment**  

## Getting Started
1. Clone the repo
2. Run 'python preprocess.py'
3. Run `streamlit run app.py`

## Dataset
[KAG_energydata_complete.csv](https://www.kaggle.com/datasets)
