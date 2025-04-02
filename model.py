import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df, features):
    # Add Sales Opportunity Patents model features
    df['Buyer_Segment_Match'] = df.apply(buyer_segment_matching, axis=1)
    df['Market_Trend_Score'] = df['Suburb'].apply(market_trend_score)
    df['Competitive_Score'] = df.apply(competitive_analysis, axis=1, df=df)
    
    df['Preferred'] = (df['Price_Per_Week'] < 1200).astype(int)  # Mock preference label
    X = df[features]
    y = df['Preferred']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def rank_properties(df, preferences, model, features):
    filtered_df = df[
        (df['Price_Per_Week'] <= preferences['Price']) &
        (df['Address'].str.contains(preferences['Address'], case=False, na=False)) &
        (df['Bedrooms'] >= preferences['Bedrooms'])
    ]
    
    if filtered_df.empty:
        return None
    
    filtered_df['Preference_Score'] = model.predict_proba(filtered_df[features])[:, 1]
    ranked_properties = filtered_df.sort_values(by='Preference_Score', ascending=False)
    
    # Add reasons for top properties
    def create_reason_string(row):
        reasons = []
        if row['Price_Per_Week'] <= preferences['Price']:
            reasons.append(f"Price: ${row['Price_Per_Week']} is within your budget.")
        if row['Balcony_Size'] > 20:
            reasons.append(f"Balcony Size: {row['Balcony_Size']} sqm, ideal for relaxation.")
        if row['Living_Area'] > 100:
            reasons.append(f"Living Area: {row['Living_Area']} sqm, providing ample space.")

        # Additional logic based on age, relationship status, and income
        if preferences['Age'] < 30 and row['Distance_to_Cafe'] < 3:
            reasons.append("Proximity to cafes, suitable for a vibrant lifestyle.")
        if preferences['Relationship_Status'] == 'Married' and row['Distance_to_School'] < 3:
            reasons.append("Close to schools, perfect for a family.")
        if preferences['Income'] > 100000 and row['Distance_to_Gym'] < 3:
            reasons.append("Nearby gyms align with your fitness goals.")
        if preferences['Income'] < 50000 and row['Price_Per_Week'] <= 800:
            reasons.append("Affordable price given your income.")
        
        return ' '.join(reasons)

    ranked_properties['Reason'] = ranked_properties.apply(create_reason_string, axis=1)
    
    return ranked_properties

# Additional functions related to Sales Opportunity Patents model
def buyer_segment_matching(row, income_threshold=80000):
    if row['Salary/Income'] > income_threshold:
        return 1  # High-income buyer segment
    return 0  # Standard buyer segment

def market_trend_score(suburb):
    # Placeholder for real market trend analysis
    trend_dict = {'SuburbA': 8, 'SuburbB': 5, 'SuburbC': 7}  # Mock trend scores
    return trend_dict.get(suburb, 5)

def competitive_analysis(row, df):
    similar_properties = df[(df['Suburb'] == row['Suburb']) & (df['Bedrooms'] == row['Bedrooms'])]
    return len(similar_properties)
