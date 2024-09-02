import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Load the models and the scaler
sc = pickle.load(open('Models/honey_standardscaler.pkl', 'rb'))
dtr = pickle.load(open('Models/honey_decisiontree.pkl', 'rb'))
gbr = pickle.load(open('Models/honey_gradeintbossing.pkl', 'rb'))
xgbgrid = pickle.load(open('Models/honey_XGBoostGrid.pkl', 'rb'))
xgboost = pickle.load(open('Models/honey_XGBoost', 'rb'))
ann = tf.keras.models.load_model('Models/honey_ann.h5')
rnn = tf.keras.models.load_model('Models/honey_rnn.h5')

# Load the dataset
df = pd.read_csv('honey_purity_dataset.csv')

# Title and introduction
st.markdown("<h2 style='text-align: center; color: yellow;'>Honey Purity Prediction App</h2>", unsafe_allow_html = True)

# Check if the plots are already created
if 'plots_created' not in st.session_state:
    st.session_state.plots_created = True

    # Create the pie chart
    fig1, ax1 = plt.subplots(figsize = (5, 5))
    pollen_analysis = df["Pollen_analysis"].value_counts().to_dict()
    ax1.pie(pollen_analysis.values(), labels = pollen_analysis.keys(), autopct = '%1.1f%%', startangle = 90)
    ax1.set_facecolor("darkgrey")
    plt.gcf().set_facecolor('darkgrey')
    ax1.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle

    # Create the bar plot
    fig2, ax2 = plt.subplots(figsize = (30, 10))
    colors = ['#ff0045', '#907c78', '#FF8C00', '#2a0600', '#d41f00', '#550c00', '#ff2500', '#9aff00', '#030000', '#2a0600']

    # Plot the performance of different regression models
    performance = {
        'Model': ['Linear Regression', 'Decision Tree', 'XGBoost', 'ADABoost', 'Gradient Boosting',
                'Linear Lasso', 'Linear GridSearch Lasso', 'XGBoost GridSearch', 'ANN Model', 'RNN Model'],
        'R2 Score': [7.35, 85.5, 97.8, 59.3, 95.3, 7.3, 5.56, 97.8, 96.5, 97.1]
    }

    performance_df = pd.DataFrame(performance)

    sns.barplot(x = 'Model', y = 'R2 Score', data = performance_df, palette = colors, ax = ax2)
    ax2.set_facecolor("darkgrey")
    plt.gcf().set_facecolor('darkgrey')
    ax2.set_xticks(range(len(performance['Model'])))
    ax2.set_xticklabels(performance['Model'], rotation=25)

    # Add labels on bars
    for index, row in performance_df.iterrows():
        ax2.text(index, row['R2 Score'], row['R2 Score'], color = 'blue', ha = "center", fontweight = 'bold', fontsize = 18)

    ax2.set_xlabel('ML Models', color = 'red', fontweight = 'bold', fontsize = 36)
    ax2.set_ylabel('Model Accuracy', color = 'red', fontweight = 'bold', fontsize = 36)
    ax2.tick_params(axis = 'x', colors = 'blue', labelsize = 24, labelrotation = 25, width = 3)
    ax2.tick_params(axis = 'y', colors = 'blue', labelsize = 24, width = 3)

    # Create the Plotly gauge chart
    model_accuracies = {
        'Gradient Boosting': 0.953,
        'XGBoost GridSearch': 0.978,
        'ANN': 0.965,
        'RNN': 0.971
    }

    fig3 = make_subplots(
        rows = 2, cols = 2,
        specs=[[{'type': 'indicator'}]*2 for _ in range(2)],
        subplot_titles = list(model_accuracies.keys())
    )

    for i, (model_name, accuracy) in enumerate(model_accuracies.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        fig3.add_trace(go.Indicator(
            mode = "gauge+number",
            value=accuracy * 100,  # Convert to percentage
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 100], 'color': "orange"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=row, col=col)

    fig3.update_layout(
        height = 600, width = 800,
        margin = dict(t=100)
    )

    # Store the plots in session state
    st.session_state.fig1 = fig1
    st.session_state.fig2 = fig2
    st.session_state.fig3 = fig3

# Display the plots from session state
st.markdown("<h4 style='text-align: center; color: blue;'>Distribution of Honey Samples by Floral Source</h4>", unsafe_allow_html = True)
st.pyplot(st.session_state.fig1)

st.markdown("<h4 style='text-align: center; color: green;'>Models Performance Comparison</h4>", unsafe_allow_html = True)
st.pyplot(st.session_state.fig2)

st.markdown("<h4 style='text-align: center; color: purple;'>Recommended Models for Prediction</h4>", unsafe_allow_html = True)
st.plotly_chart(st.session_state.fig3)


# Prediction Section
st.write("Predict the purity of Honey based on specific features.")

# Input features
Color_Score = st.slider("Color Score of Honey sample", 1.0, 10.0)
Density = st.slider("Density of Honey sample at 25°C", 1.21, 1.86)
Water_Content = st.slider("Water Content in Honey sample", 12.0, 25.0)
pH = st.slider("pH level of Honey sample", 2.50, 7.50)
Electrical_Conductivity = st.slider("Electrical Conductivity of Honey sample (milliSiemens per centimeter)", 0.75, 0.9)
Fructose_Level = st.slider("Fructose Level of Honey sample",  20.0, 50.0)
Glucose_Level = st.slider("Glucose Level of Honey sample", 20.0, 45.0)
Viscosity = st.number_input("Viscosity of Honey sample in centipoise (1500 to 10000)", min_value = 1500, max_value = 10000, step = 100)
Pollen_Analysis = st.selectbox("Floral Source of Honey sample",
                                   ["Clover", "Wildflower", "Orange Blossom", "Alfalfa", "Acacia", "Lavender", "Eucalyptus", 
                                    "Buckwheat", "Manuka", "Sage", "Sunflower", "Borage", "Rosemary", "Thyme", 
                                    "Heather", "Tupelo", "Blueberry", "Chestnut", "Avocado"])

# Collect all numerical input features
num_features = [Color_Score, Density, Water_Content, pH,
       Electrical_Conductivity, Fructose_Level, Glucose_Level,
       Viscosity]
num_features_cols = ['Color Score', 'Density', 'Water Content', 'pH',
       'Electrical Conductivity', 'Fructose Level', 'Glucose Level',
       'Viscosity']

num_features_df = pd.DataFrame([num_features], columns=num_features_cols)
sc_num_features = sc.transform(num_features_df)

# Collect all categorical input features
cat_features=[
        1 if Pollen_Analysis == 'Clover' else 0,
        1 if Pollen_Analysis == 'Wildflower' else 0,
        1 if Pollen_Analysis == 'Orange Blossom' else 0,
        1 if Pollen_Analysis == 'Alfalfa' else 0,
        1 if Pollen_Analysis == 'Acacia' else 0,
        1 if Pollen_Analysis == 'Lavender' else 0,
        1 if Pollen_Analysis == 'Eucalyptus' else 0,
        1 if Pollen_Analysis == 'Buckwheat' else 0,
        1 if Pollen_Analysis == 'Manuka' else 0,
        1 if Pollen_Analysis == 'Sage' else 0,
        1 if Pollen_Analysis == 'Sunflower' else 0,
        1 if Pollen_Analysis == 'Borage' else 0,
        1 if Pollen_Analysis == 'Rosemary' else 0,
        1 if Pollen_Analysis == 'Thyme' else 0,
        1 if Pollen_Analysis == 'Heather' else 0,
        1 if Pollen_Analysis == 'Tupelo' else 0,
        1 if Pollen_Analysis == 'Blueberry' else 0,
        1 if Pollen_Analysis == 'Chestnut' else 0,
        1 if Pollen_Analysis == 'Avocado' else 0
    ]

# Ensure column names match the features
cols = ['Color Score', 'Density', 'Water Content', 'pH',
           'Electrical Conductivity', 'Fructose Level', 'Glucose Level',
           'Viscosity', 'Pollen Analysis_Acacia', 'Pollen Analysis_Alfalfa',
           'Pollen Analysis_Avocado', 'Pollen Analysis_Blueberry',
           'Pollen Analysis_Borage', 'Pollen Analysis_Buckwheat',
           'Pollen Analysis_Chestnut', 'Pollen Analysis_Clover',
           'Pollen Analysis_Eucalyptus', 'Pollen Analysis_Heather',
           'Pollen Analysis_Lavender', 'Pollen Analysis_Manuka',
           'Pollen Analysis_Orange Blossom', 'Pollen Analysis_Rosemary',
           'Pollen Analysis_Sage', 'Pollen Analysis_Sunflower',
           'Pollen Analysis_Thyme', 'Pollen Analysis_Tupelo',
           'Pollen Analysis_Wildflower'
    ]

# Combine all features into one array
features = np.concatenate([sc_num_features, np.array(cat_features).reshape(1, -1)], axis = 1)
features_df = pd.DataFrame(features, columns = cols)

# Model selection
model_choice = st.selectbox("Choose a model", 
                                ["Decision Tree", "Gradient Boosting", "XGBoost", "XGBoost GridSearch", "ANN", "RNN"])

# Prediction
if 'prediction' not in st.session_state or st.button("Predict Honey Purity"):
    tf.keras.backend.clear_session()
    
    if model_choice == 'Decision Tree':
        st.session_state.prediction = dtr.predict(features_df)[0]
    elif model_choice == 'Gradient Boosting':
        st.session_state.prediction = gbr.predict(features_df)[0]
    elif model_choice == 'XGBoost':
        st.session_state.prediction = xgboost.predict(features_df)[0]
    elif model_choice == 'XGBoost GridSearch':
        st.session_state.prediction = xgbgrid.predict(features_df)[0]
    elif model_choice == 'ANN':
        st.session_state.prediction = ann.predict(features_df)[0]
    elif model_choice == 'RNN':
        st.session_state.prediction = rnn.predict(features_df)[0]

# Display the prediction
prediction_value = st.session_state.prediction.item()

thumbs_up = "👍"
thumbs_down = "👎"

st.write(f"Model Selected :  {model_choice}")
st.write(f"Model Accuracy :  {prediction_value:.3f}")
st.write(
    f"Predicted Honey Purity :  {'Honey is Pure ' + thumbs_up if prediction_value >= 0.8 else 'Honey is Not Pure ' + thumbs_down}"
)

