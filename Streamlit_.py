import streamlit as st
import joblib
import pandas as pd

# Load the DataFrame
df = pd.read_csv('Cars_Dataset.csv')

# Load the pickled files
rf_model = joblib.load('random_forest_model.pkl')
label_encoded = joblib.load('label_encoding.pkl')
onehot_encoded = joblib.load('onehot_encoding.pkl')
scaler = joblib.load('scaling.pkl')
onehot_encoded_columns = joblib.load('onehot_columns.pkl')  # If you have this pickled as well

    
# Setting webpage

st.set_page_config(layout="centered")

st.title(':green[USED CAR PRICE PREDICTION ]')

st.image("FPQkX1jaUAQKpYb.jpg")

st.sidebar.image("logo.png")

st.sidebar.title(':green[Enter the Following Details]')


# Sidebar input fields

city = st.sidebar.selectbox('City', sorted(df['City'].unique()))

transmission = st.sidebar.selectbox('Transmission', sorted(df['transmission'].unique()))

oem = st.sidebar.selectbox('OEM', sorted(df['oem'].unique()))

model_year = st.sidebar.number_input('Model Year', min_value=int(df['modelYear'].min()), max_value=int(df['modelYear'].max()), step=1)

fuel_type = st.sidebar.selectbox('Fuel Type', sorted(df['Fuel Type'].unique()))

owner_no = st.sidebar.number_input('Owner Number', min_value=int(df['ownerNo'].min()), max_value=int(df['ownerNo'].max()), step=1)

km = st.sidebar.number_input('Kilometers Driven', min_value=int(df['km'].min()), max_value=int(df['km'].max()), step=1000)

mileage = st.sidebar.number_input('Mileage', min_value=float(df['Mileage'].min()), max_value=float(df['Mileage'].max()), step=0.1)

bodytype = st.sidebar.selectbox('Body Type', sorted(df['bt'].unique()))

insurance = st.sidebar.selectbox('Insurance Validity', sorted(df['Insurance Validity'].unique()))

steering = st.sidebar.selectbox('Steering Type', sorted(df['Steering Type'].unique()))

seats = st.sidebar.number_input('Seaters', min_value=int(df['Seats_specs'].min()), max_value=int(df['Seats_specs'].max()), step=1)

Tyre = st.sidebar.selectbox('Tyre Type', sorted(df['Tyre Type'].unique()))

Trunck_space = st.sidebar.number_input('Trunk space(litres)', min_value=float(df['Cargo Volumn'].min()), max_value=float(df['Cargo Volumn'].max()), step=0.50)

# Prepare input data for prediction
input_data = {
    'City': city,
    'transmission': transmission,
    'oem': oem,
    'ownerNo': owner_no,
    'modelYear': model_year,
    'Fuel Type': fuel_type,
    'km': km,
    'bt': bodytype,
    'Insurance Validity': insurance,
    'Steering Type': steering,
    'Mileage': mileage,
    'Seats_specs' : seats,
    'Tyre Type' : Tyre,
    'Cargo Volumn' : Trunck_space
}
input_df = pd.DataFrame([input_data])

# Label encoding

# Apply pre-trained label encoders to transform specified categorical columns into numerical form.

label = input_df[['oem', 'Fuel Type', 'Tyre Type', 'Steering Type']].copy()
for col1 in label.columns:
    if col1 in label_encoded:
        label[col1] = label_encoded[col1].transform(label[col1])

# One-Hot encoding

# setting the relevant columns to 1 based on input values keep only the columns with non-zero values.

onehot = pd.DataFrame(0, index=[0], columns=onehot_encoded_columns)
for col2 in ['transmission', 'bt', 'Insurance Validity', 'City']:
    col_val = input_df[col2].iloc[0]
    col_name = f'{col2}_{col_val}'
    if col_name in onehot.columns:
        onehot[col_name] = 1
onehot = onehot.loc[:, (onehot != 0).any(axis=0)]

# Concatenate label and one-hot encoded data
encoded = pd.concat([label, onehot], axis=1)

# Scaling
Scale = input_df[['ownerNo', 'km', 'Mileage', 'modelYear', 'Cargo Volumn', 'Seats_specs']].copy()
for col3 in Scale.columns:
    if col3 in scaler:
        Scale[col3] = scaler[col3].transform(Scale[[col3]])

# Final concatenated DataFrame for prediction
final_input = pd.concat([Scale, encoded], axis=1)

# Ensure all columns are in the final input in the correct order
original_features = rf_model.feature_names_in_
final_input = final_input.reindex(columns=original_features, fill_value=0)
predict_button = st.sidebar.button("Predict Price")

if predict_button:
        # Prediction
    predict_price = rf_model.predict(final_input)
    st.success(f"The predicted price is: ₹{predict_price[0]:,.2f}")