import streamlit as st
import pandas as pd
from model import train_model, rank_properties

# Specify the path to the CSV file
file_path = 'C:/Users/savan/Downloads/nsw_property_data.csv'

# Load data
df = pd.read_csv(file_path)

# Define features for the model, including new features from the Sales Opportunity Patents model
features = ['Price_Per_Week', 'Bedrooms', 'Carpark', 'Balcony_Size', 'Living_Area',
            'Distance_to_Parks', 'Distance_to_Schools', 'Distance_to_Shopping_Centers',
            'Distance_to_Public_Transport', 'Distance_to_Hospitals',
            'Buyer_Segment_Match', 'Market_Trend_Score', 'Competitive_Score']

# Train AI model on the data
model = train_model(df, features)

def main():
    st.title("AI Property Comparison with Enhanced Sales Prediction")

    # User Preferences
    st.sidebar.header("User Preferences")
    with st.form("user_preferences_form"):
        address = st.text_input('Enter Suburb:')
        bedrooms = st.number_input('Number of Bedrooms:', min_value=1, max_value=5, value=2)
        max_price = st.number_input('Maximum Price Per Week:', min_value=100, max_value=2000, value=1000)
        age = st.number_input('Your Age:', min_value=18, max_value=100, value=30)
        relationship_status = st.selectbox('Relationship Status:', options=['Single', 'Married'])
        income = st.number_input('Your Income:', min_value=20000, max_value=200000, value=50000)
        
        preferences = {
            'Price': max_price,
            'Address': address,
            'Bedrooms': bedrooms,
            'Age': age,
            'Relationship_Status': relationship_status,
            'Income': income
        }
        
        submit_button = st.form_submit_button(label="Show Properties")
    
    if submit_button:
        # Filter properties
        filtered_properties = df[
            (df['Price_Per_Week'] <= preferences['Price']) &
            (df['Address'].str.contains(preferences['Address'], case=False, na=False)) &
            (df['Bedrooms'] >= preferences['Bedrooms'])
        ]
        st.subheader("Properties Matching Your Criteria")
        st.table(filtered_properties[['Address', 'Price_Per_Week', 'Bedrooms', 'Carpark']])
        
        # Rank and display top properties
        top_properties = rank_properties(filtered_properties, preferences, model, features)
        
        if top_properties is not None:
            st.subheader("Top Properties Based on Additional Factors")
            for idx, property in top_properties.head(2).iterrows():
                st.write(f"### **Address:** {property['Address']}")
                st.image(property['Image_URL'], caption=property['Address'], use_column_width=True)
                st.write(f"**Price Per Week:** ${property['Price_Per_Week']}")
                st.write(f"**Bedrooms:** {property['Bedrooms']}")
                st.write(f"**Carparks:** {property['Carpark']}")
                st.write(f"**Balcony Size:** {property['Balcony_Size']} sqm")
                st.write(f"**Living Area:** {property['Living_Area']} sqm")
                
                st.write("### **Nearby Amenities:**")
                st.write(f"Distance to Park: {property['Distance_to_Parks']} km")
                st.write(f"Distance to School: {property['Distance_to_Schools']} km")
                st.write(f"Distance to Shopping Centre: {property['Distance_to_Shopping_Centers']} km")
                st.write(f"Distance to Public Transport: {property['Distance_to_Public_Transport']} km")
                st.write(f"Distance to Hospital: {property['Distance_to_Hospitals']} km")
                
                st.write("### **Why This Property is Great:**")
                st.write(property['Reason'])

if __name__ == "__main__":
    main()
