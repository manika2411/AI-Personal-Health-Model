import pandas as pd
import numpy as np
import csv
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Main program loop
running = True
while running:
    print("\nWelcome to AI Personal Assistant")
    print("===============================")
    print("1. Sleep Quality Prediction")
    print("2. Nutrition Recommendation")
    print("3. Symptom to Disease Prediction")
    print("4. Exit")

    model_choice = input("\nSelect a model (1-4): ")

    # Sleep Model
    if model_choice == "1":
        print("\nRunning Sleep Quality Prediction Model...")
        data = pd.read_csv(r"C:\Users\DELL\Downloads\Sleep_health_and_lifestyle_dataset.csv")
        # Fill missing values
        for col in ['Sleep Duration', 'Heart Rate', 'Physical Activity Level', 'Quality of Sleep']:
            data[col] = data[col].fillna(0)
        # Features and labels
        x = data[['Sleep Duration', 'Heart Rate', 'Physical Activity Level']]
        y = (data['Quality of Sleep'] >= 6).astype(int)  # Good sleep if quality >=6
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Model
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        # Evaluate
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")
        # Feature importance
        coefficients = pd.DataFrame({
            'Feature': x.columns,
            'Coefficient': model.coef_[0]
        })
        print("\nFeature importance:")
        print(coefficients.sort_values('Coefficient', ascending=False))
        # Prediction
        try:
            print("\nPredict your sleep quality:")
            d = float(input("Enter sleep duration (hours): "))
            h = int(input("Enter average heart rate: "))
            a = int(input("Enter physical activity level (minutes): "))
            
            # Create a DataFrame instead of list to match feature names
            input_df = pd.DataFrame([[d, h, a]], columns=x.columns)
            
            p = model.predict(input_df)
            prob = model.predict_proba(input_df)[0][1]
            
            print(f"\nPrediction: {'Good Sleep Quality' if p[0] == 1 else 'Poor Sleep Quality'}")
            print(f"Confidence: {prob:.2f} ({prob*100:.1f}%)")
            
            if p[0] == 1:
                print("\nFactors that contributed to good sleep prediction:")
            else:
                print("\nSuggestions to improve sleep quality:")   
            
            if d < 7:
                print(f"- Sleep duration ({d} hours) could be improved. 7-9 hours is recommended.")
            if a < 30:
                print(f"- Physical activity level ({a} minutes) is low. Try to get at least 30 minutes daily.")
            if h > 80:
                print(f"- Heart rate ({h} bpm) is elevated. Relaxation techniques might help.")    
        except ValueError:
            print("Please enter valid numbers for all inputs.")
        print("Thank you for using Sleep and Physical Activity Recommendation System!")

    # Nutrition Model
    elif model_choice == "2":
        print("\nRunning Nutrition Recommendation Model...")
        # Load data
        file_path = r"C:\Users\DELL\Downloads\nutrition.csv"
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['calories', 'carbohydrate', 'protein', 'fat'], how='all')
        # Extract features
        features = df[['calories', 'carbohydrate', 'protein', 'fat']].copy()
        features.columns = ['Calories', 'Carbohydrate (g)', 'Protein (g)', 'Total lipid (fat) (g)']
        # Clean and convert data
        for col in features.columns:
            features[col] = features[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
            features[col] = pd.to_numeric(features[col], errors='coerce')
        features = features.dropna()
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        # Create KNN model
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(scaled_features)
        # User input for recommendation type
        print("\nNutrition Recommendation System")
        print("==============================")
        print("\nOptions:")
        print("1. Find foods similar to specific nutritional values")
        print("2. Find low-carb foods")
        print("3. Find high-protein foods")
        choice = input("\nEnter your choice (1-3): ")
        # Process selected option
        if choice == '1':
            calories = float(input("Enter desired calories: "))
            carbs = float(input("Enter desired carbohydrates (g): "))
            protein = float(input("Enter desired protein (g): "))
            fat = float(input("Enter desired fat (g): "))
            query = [calories, carbs, protein, fat]
            
        elif choice == '2':
            query = [300, 5, 20, 10]  # Low-carb profile
            print("\nFinding Low-Carb Food Recommendations:")
            
        elif choice == '3':
            query = [300, 15, 30, 5]  # High-protein profile
            print("\nFinding High-Protein Food Recommendations:")
            
        else:
            print("Invalid choice.")
            continue
        # Get recommendations for options 1-3
        query_df = pd.DataFrame([query], columns=features.columns)   
        scaled_query = scaler.transform(query_df)
        distances, indices = knn.kneighbors(scaled_query, n_neighbors=5)
        # Display recommendations
        food_names = df['name'] if 'name' in df.columns else df.iloc[:, 0]
        print("\nTop 5 Food Recommendations:\n")
        for i, idx in enumerate(indices[0]):
            name = food_names.iloc[idx]
            calories = features.iloc[idx]['Calories']
            carbs = features.iloc[idx]['Carbohydrate (g)']
            protein = features.iloc[idx]['Protein (g)']
            fat = features.iloc[idx]['Total lipid (fat) (g)']
            
            print(f"{i+1}. {name} - Calories: {calories:.1f}, Carbs: {carbs:.1f}g, Protein: {protein:.1f}g, Fat: {fat:.1f}g")
        print("Thank you for using the Nutrition Recommendation System!")

    # Symptom to Disease Model
    elif model_choice == "3":
        print("\nRunning Symptom to Disease Prediction Model...")
        # Load data from CSV and build the symptom-disease tree
        symptom_tree = {}
        file = open(r"C:\Users\DELL\Downloads\Disease_symptom_and_patient_profile_dataset.csv")
        reader = csv.DictReader(file)
        for row in reader:
            # Create a symptom key based on Yes/No values
            symptom_key = f"Fever:{row['Fever']},Cough:{row['Cough']},Fatigue:{row['Fatigue']},Difficulty Breathing:{row['Difficulty Breathing']}"
            
            # Initialize the symptom combination if not exists
            if symptom_key not in symptom_tree:
                symptom_tree[symptom_key] = []
            
            # Add the disease to this symptom combination
            symptom_tree[symptom_key].append(row['Disease'])
        file.close()
        # Count and sort diseases for each symptom combination
        for key in symptom_tree:
            diseases_count = Counter(symptom_tree[key])
            symptom_tree[key] = diseases_count.most_common(5)  # Keep top 5 diseases
        # Main program
        print("Disease Prediction System")
        print("========================")
        print("\nPlease enter your symptoms (Yes/No):")
        # Get symptoms from user
        fever = input("Do you have fever? (Yes/No): ").strip().capitalize()
        cough = input("Do you have cough? (Yes/No): ").strip().capitalize()
        fatigue = input("Do you feel fatigue? (Yes/No): ").strip().capitalize()
        difficulty_breathing = input("Do you have difficulty breathing? (Yes/No): ").strip().capitalize()
        # Create the symptom key
        symptom_key = f"Fever:{fever},Cough:{cough},Fatigue:{fatigue},Difficulty Breathing:{difficulty_breathing}"
        # Predict diseases based on symptoms
        if symptom_key in symptom_tree:
            predictions = symptom_tree[symptom_key]
            
            print("\nPossible diseases based on your symptoms:")
            print("----------------------------------------")
            for i, (disease, count) in enumerate(predictions, 1):
                print(f"{i}. {disease} (Count: {count})")
        else:
            print("\nNo matching diseases found for this symptom combination.")
        print("Thank you for using the Disease Prediction System!")

    # Exit option
    elif model_choice == "4":
        print("Thank you for using AI Personal Assistant. Goodbye!")
        running = False

    else:
        print("Invalid choice. Please select a number from 1 to 4.")