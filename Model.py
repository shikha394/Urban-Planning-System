import base64
import json
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingRegressor, \
    RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

import pickle

import warnings

warnings.filterwarnings("ignore")


def city_list_gen(df, pollutant_name):
    df['CITIES'] = df['CITIES'].apply(lambda row: row.lower())
    lists = df['CITIES'].unique().tolist()
    with open(f'{pollutant_name}.json', 'w', encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False, indent=4)
    return lists, df


def selecting_city(df, city):
    df_selected = df.loc[df['CITIES'] == city.lower()].copy()
    df_selected.drop(['CITIES'], axis=1, inplace=True)
    df_selected = df_selected.T
    df_selected.dropna(inplace=True)
    df_selected = df_selected.reset_index()
    return df_selected


def prepare_data(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        end_ix = i + sequence_length
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def fit_sarima_model(train_data, order, seasonal_order):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    return results


def prediction_sarima(model, data, sequence_length, scaler):
    input_sequence = data[-sequence_length:]
    input_sequence = np.reshape(input_sequence, (sequence_length, data.shape[1]))
    predicted_normalized = model.predict(start=model.nobs, end=model.nobs + sequence_length - 1, exog=input_sequence)
    predicted_population = scaler.inverse_transform(predicted_normalized.reshape(-1, 1))
    return int(predicted_population[-1][0])


def prediction_model(df):
    x = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model


def plot_trend(df_selected, model, sequence_length, scaler, city, pollutant_name, model_type):
    data = scaler.transform(df_selected.iloc[:, 1].values.reshape(-1, 1))
    X, _ = prepare_data(data, sequence_length)

    plt.plot(df_selected.iloc[:, 0].astype(int).values, df_selected.iloc[:, 1].values,
             label=f'{pollutant_name} - Trend in Data', marker='o')

    for i in range(len(X)):
        if i == len(X) - 1:
            x_values = range(int(df_selected.iloc[:, 0].values[i + 1]),
                             int(df_selected.iloc[:, 0].values[i + 1]) + sequence_length)
            input_sequence = X[i].reshape(sequence_length, -1)

            if model_type == 'sarima':
                prediction_result = prediction_sarima(model, input_sequence, sequence_length, scaler)


def combine_models(city, year, pollutant_name=None):
    result_dict = {}  # Store results in a dictionary
    city = city.lower()

    # Model Population
    df = pd.read_csv('populationpp.csv')

    # Basic preprocessing
    X = df.drop('population', axis=1)
    y = df['population']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_features = ['year']
    categorical_features = ['city']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    # Base models
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor()
    xgb_model = XGBRegressor()
    catboost_model = CatBoostRegressor(silent=True)
    nn_model = MLPRegressor(max_iter=1000)

    # Stacking ensemble model
    stacked_model = StackingRegressor(
        estimators=[
            ('linear', linear_model),
            ('random_forest', rf_model),
            ('xgboost', xgb_model),
            ('catboost', catboost_model),
            ('neural_network', nn_model)
        ],
        final_estimator=LinearRegression()
    )
    # Create and evaluate each base model separately
    models = [linear_model, rf_model, xgb_model, catboost_model, nn_model, stacked_model]
    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'CatBoost', 'Neural Network', 'Stacked Ensemble']
    results = []

    for model, name in zip(models, model_names):
        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results.append({'Model': name, 'MSE': mse})

    results_df = pd.DataFrame(results)
    # Save the best model
    best_model_index = results_df['MSE'].idxmin()
    best_model = models[best_model_index]
    dump(best_model, 'best_model.joblib')
    # print(f"\nBest Model ({model_names[best_model_index]}) saved as 'best_model.joblib'")

    # Predict using the saved model
    loaded_model = load('best_model.joblib')

    # Take user input for prediction
    input_city = city.capitalize()
    input_year = year

    # Check if the input city is in the original dataset
    if input_city not in df['city'].values:
        print(f"City '{input_city}' is not available in the original dataset.")
    else:
        # Filter data for the input city
        city_data = df[df['city'] == input_city].sort_values('year')
        # print(city_data)

        # Preprocess data for plotting
        X_city = city_data.drop('population', axis=1)
        # X_city['year'] = input_year  # Set the input year
        # X_city[input_city] = input_year  # Set the input year
        new_row = pd.DataFrame({'city': input_city, 'year': input_year}, index=[0])
        X_city = pd.concat([X_city.loc[:], new_row]).reset_index(drop=True)
        X_city_preprocessed = preprocessor.transform(X_city)

        # Make predictions for the input city and year
        predicted_population = loaded_model.predict(X_city_preprocessed)
        # print(predicted_population)
        print(f"Predicted population for {input_city} for {input_year} is {int(predicted_population[4])}")

        prediction_years = pd.concat([city_data['year'], pd.DataFrame({input_year}, index=[0])]).reset_index(drop=True)
        # print(prediction_years)
        result_dict['Population Prediction'] = predicted_population[4]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(city_data['year'], city_data['population'], label='Actual Population', marker='o')
        plt.plot(prediction_years, predicted_population, label='Predicted Population', marker='x')
        plt.scatter(input_year, predicted_population[4], color='red', label=f'Predicted Population of {input_year}',
                    marker='x', s=100)
        plt.title(f'Actual vs Predicted Population for {input_city} in {input_year}')
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.legend()
        # plt.show()

        # Save the plot to a BytesIO object
        img_buffer1 = BytesIO()
        plt.savefig(img_buffer1, format='png')
        img_buffer1.seek(0)  # Move the cursor to the beginning of the buffer

        # Debug information
        print(f"Image buffer size: {img_buffer1.getbuffer().nbytes} bytes")  # Print the size of the buffer

        # Ensure that the image content is not empty
        if img_buffer1.getbuffer().nbytes > 0:
            img_data = base64.b64encode(img_buffer1.getvalue()).decode("utf-8")
        else:
            img_data = None

        img_buffer1.close()  # Close the buffer after reading its content

        # Debug information
        print(f"Encoded image data: {img_data[:50]}...")  # Print the first 50 characters of the encoded data

        # Close the plot
        plt.close()

    # Model 1 (SO2)
    df1 = pd.read_csv('SO2.csv')
    lists1, df_so2 = city_list_gen(df1, 'SO2')
    result_so2 = None
    if city in lists1:
        df_selected_so2 = selecting_city(df_so2, city)
        if not df_selected_so2.empty:
            scaler_so2 = MinMaxScaler()
            data_so2 = scaler_so2.fit_transform(df_selected_so2.iloc[:, 1].values.reshape(-1, 1))
            sequence_length_so2 = 3
            X_so2, y_so2 = prepare_data(data_so2, sequence_length_so2)
            X_train_so2, _, y_train_so2, _ = train_test_split(X_so2, y_so2, test_size=0.2, random_state=42)
            order_so2 = (1, 1, 1)
            seasonal_order_so2 = (1, 1, 1, 12)
            model_so2 = fit_sarima_model(y_train_so2, order_so2, seasonal_order_so2)
            result_so2 = prediction_sarima(model_so2, data_so2, sequence_length_so2, scaler_so2)
            print(f"Result (SO2): {city.upper()} SO2 level in {year} will be {result_so2:,d}")
            plot_trend(df_selected_so2, model_so2, sequence_length_so2, scaler_so2, city, 'SO2', 'sarima')
            result_dict['SO2'] = result_so2
        else:
            result_dict['SO2'] = 'Data not available for SO2'
            print('Data not available for SO2.')
    else:
        result_dict['SO2'] = 'Invalid city name for SO2'
        print('Kindly check city name spelling for SO2.')

    # Model 2 (PM10)
    df2 = pd.read_csv('PM10.csv')
    lists2, df_pm10 = city_list_gen(df2, 'PM10')
    result_pm10 = None
    if city in lists2:
        df_selected_pm10 = selecting_city(df_pm10, city)
        if not df_selected_pm10.empty:
            scaler_pm10 = MinMaxScaler()
            data_pm10 = scaler_pm10.fit_transform(df_selected_pm10.iloc[:, 1].values.reshape(-1, 1))
            sequence_length_pm10 = 3
            X_pm10, y_pm10 = prepare_data(data_pm10, sequence_length_pm10)
            X_train_pm10, _, y_train_pm10, _ = train_test_split(X_pm10, y_pm10, test_size=0.2, random_state=42)
            order_pm10 = (1, 1, 1)
            seasonal_order_pm10 = (1, 1, 1, 12)
            model_pm10 = fit_sarima_model(y_train_pm10, order_pm10, seasonal_order_pm10)
            result_pm10 = prediction_sarima(model_pm10, data_pm10, sequence_length_pm10, scaler_pm10)
            print(f"Result (PM10): {city.upper()} PM10 level in {year} will be {result_pm10:,d}")
            plot_trend(df_selected_pm10, model_pm10, sequence_length_pm10, scaler_pm10, city, 'PM10', 'sarima')
            result_dict['PM10'] = result_pm10
        else:
            result_dict['PM10'] = 'Data not available for PM10'
            print('Data not available for PM10.')
    else:
        result_dict['PM10'] = 'Invalid city name for PM10'
        print('Kindly check city name spelling for PM10.')

    # Model 3 (NO2)
    df3 = pd.read_csv('NO2.csv')
    lists3, df_no2 = city_list_gen(df3, 'NO2')
    if city in lists3:
        df_selected_no2 = selecting_city(df_no2, city)
        scaler_no2 = MinMaxScaler()
        data_no2 = scaler_no2.fit_transform(df_selected_no2.iloc[:, 1].values.reshape(-1, 1))
        sequence_length_no2 = 3
        X_no2, y_no2 = prepare_data(data_no2, sequence_length_no2)
        X_train_no2, _, y_train_no2, _ = train_test_split(X_no2, y_no2, test_size=0.2, random_state=42)
        order_no2 = (1, 1, 1)
        seasonal_order_no2 = (1, 1, 1, 12)
        model_no2 = fit_sarima_model(y_train_no2, order_no2, seasonal_order_no2)
        result_no2 = prediction_sarima(model_no2, data_no2, sequence_length_no2, scaler_no2)
        print(f"Result (NO2): {city.upper()} NO2 level in {year} will be {result_no2:,d}")
        plot_trend(df_selected_no2, model_no2, sequence_length_no2, scaler_no2, city, 'NO2', 'sarima')
        result_dict['NO2'] = result_no2
    else:
        result_dict['NO2'] = 'Data not available forNO2'
        print('Data not available for NO2.')
    # else:
    #     result_dict['NO2'] = 'Invalid city name for PM10'
    #     print('Kindly check city name spelling for PM10.')

    # Plotting Combined Results
    plt.legend()  # Add legend to the plot
    plt.title(f'Pollutant Level Trend Over the Years for {city.upper()}')
    plt.xlabel('Year')
    plt.ylabel('Pollutant Level')

    # Save the plot to a BytesIO object
    img_buffer2 = BytesIO()
    plt.savefig(img_buffer2, format='png')
    img_buffer2.seek(0)  # Move the cursor to the beginning of the buffer
    img_pop: str = base64.b64encode(img_buffer2.read()).decode("utf-8")
    img_buffer2.close()  # Close the buffer after reading its content

    # Close the plot
    plt.close()

    # Combine predictions for the second model
    input_for_second_model = {
        'PM10': result_pm10,
        'NO2': result_no2,
        'SO2': result_so2
    }

    # Save the input_for_second_model dictionary
    with open('input_for_second_model.pkl', 'wb') as file:
        pickle.dump(input_for_second_model, file)

    # Load the input_for_second_model dictionary
    with open('input_for_second_model.pkl', 'rb') as file:
        input_for_second_model = pickle.load(file)

    # Load the second model and make predictions
    le = LabelEncoder()
    df_second_model = pd.read_csv('airQuality.csv')
    # Replace with the actual dataset used for the second model
    df_second_model['AQI_Bucket'] = le.fit_transform(df_second_model['AQI_Bucket'])
    df_second_model.dropna(subset=['AQI_Bucket'], inplace=True)
    df_second_model.dropna(subset=['PM10'], inplace=True)
    df_second_model.dropna(subset=['NO2'], inplace=True)
    df_second_model.dropna(subset=['SO2'], inplace=True)

    X_second_model = df_second_model[['PM10', 'NO2', 'SO2']]
    y_second_model = df_second_model['AQI_Bucket']

    # Handle NaN values in the input data
    X_second_model.fillna(0, inplace=True)  # Replace NaN values with 0 or choose an appropriate strategy

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_second_model, y_second_model)
    gb_classifier.fit(X_second_model, y_second_model)

    # Create a voting classifier with the three base classifiers
    voting_classifier = VotingClassifier(estimators=[
        ('random_forest', rf_classifier),
        ('gradient_boosting', gb_classifier)
    ], voting='hard')

    # Train the voting classifier
    voting_classifier.fit(X_second_model, y_second_model)

    # Handle NaN values in the input for second model
    for key, value in input_for_second_model.items():
        if pd.isna(value):
            input_for_second_model[key] = 0  # Replace NaN values with 0 or choose an appropriate strategy

    # Make predictions using the ensemble classifier
    ensemble_user_pred = voting_classifier.predict([list(input_for_second_model.values())])

    # Decode the predicted labels back to original class names
    ensemble_pred_class = le.inverse_transform(ensemble_user_pred)[0]

    # Return the results in a dictionary
    result_dict['Ensemble Prediction'] = ensemble_pred_class

    return result_dict, img_data, img_pop


# Run the code
if __name__ == "__main__":
    city_input = input("Please input the city name: ").lower()
    year_input = int(input("Please input the year to predict: "))
    result = combine_models(city_input, year_input)
    # print(result)  # This line will print the complete results dictionary
