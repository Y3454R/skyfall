import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Factory to create different components
class ComponentFactory:
    @staticmethod
    def create_data_loader(file_path):
        return DataLoader(file_path)

    @staticmethod
    def create_data_preprocessor():
        return DataPreprocessor()

    @staticmethod
    def create_model_builder():
        return ModelBuilder()

    @staticmethod
    def create_trainer(model):
        return Trainer(model)

    @staticmethod
    def create_forecaster(model):
        return Forecaster(model)


# New Class: DataLoader
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_preprocess_data(self):
        """
        Load and preprocess data for training and testing.
        """
        # Load data
        df = pd.read_csv(self.file_path)

        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Set Date column as index
        df.set_index('Date', inplace=True)

        # Handle missing values
        df.fillna(method='ffill', inplace=True)  # Forward fill for missing values

        # Resample data to ensure consistency (e.g., daily frequency)
        df = df.resample('D').mean()  # Or use 'M' for monthly, 'W' for weekly

        # Scale data
        scaler = StandardScaler()
        df[['minTemp', 'Rainfall']] = scaler.fit_transform(df[['minTemp', 'Rainfall']])

        # Create lag features
        df['minTemp_lag_1'] = df['minTemp'].shift(1)
        df['RainFall_lag_1'] = df['Rainfall'].shift(1)

        # Drop rows with NaN values created by shifting
        df.dropna(inplace=True)

        # Split data into train and test sets
        train = df.iloc[:int(0.8 * len(df))]
        test = df.iloc[int(0.8 * len(df)):]

        # Define features and target variable
        X_train = train[['minTemp_lag_1', 'RainFall_lag_1']]
        y_train = train['Rainfall']
        X_test = test[['minTemp_lag_1', 'RainFall_lag_1']]
        y_test = test['Rainfall']

        return X_train, X_test, y_train, y_test, df


# Step 2: Data Preprocessor
class DataPreprocessor:
    def prepare_data(self, X_train, X_test):
        """
        Reshape data for LSTM input: [samples, time steps, features].
        """
        X_train_reshaped = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
        return X_train_reshaped, X_test_reshaped


# Step 3: Model Builder
class ModelBuilder:
    def build_lstm_model(self, input_shape):
        """
        Build and compile an LSTM model.
        """
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model


# Step 4: Trainer
class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, epochs, validation_split=0.2):
        """
        Train the model with optional validation split.
        """
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=1)
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using Mean Squared Error (MSE).
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return mse, y_pred

    @staticmethod
    def plot_loss(history):
        """
        Plot training and validation loss over epochs.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


# Step 5: Forecaster
class Forecaster:
    def __init__(self, model):
        self.model = model

    def forecast(self, last_values, future_dates, update_last_values):
        future_predictions = []
        for _ in range(len(future_dates)):
            # Reshape last_values to (samples=1, timesteps=1, features=2)
            reshaped_last_values = last_values.reshape((1, 1, last_values.shape[1]))
            future_pred = self.model.predict(reshaped_last_values)
            future_predictions.append(future_pred[0][0])  # Extract scalar value
            # Update last_values with the new prediction
            last_values = update_last_values(last_values, future_pred[0][0])
        return future_predictions

    @staticmethod
    def visualize_forecast(df, forecast_df):
        """
        Plot historical and forecasted rainfall.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Rainfall'], color='blue', label='Historical Rainfall')
        plt.plot(forecast_df.index, forecast_df['Forecasted_Rainfall'], color='red', linestyle='--', label='Forecasted Rainfall')
        plt.xlabel('Date')
        plt.ylabel('Rainfall')
        plt.title('Historical vs Forecasted Rainfall')
        plt.legend()
        plt.show()


# Step 6: Feature Update Function
def update_last_values(last_values, new_prediction):
    # Update last_values with the new prediction for sequential forecasting
    return np.array([[last_values[0][1], new_prediction]])



# Step 7: Main Execution
if __name__ == "__main__":
    # Specify the file path for the data
    file_path = 'dataset/raw_dataset/merged_data_timeserie.csv'

    # Load and preprocess data
    data_loader = ComponentFactory.create_data_loader(file_path)
    X_train, X_test, y_train, y_test, df = data_loader.load_and_preprocess_data()

    # Create components using the factory
    preprocessor = ComponentFactory.create_data_preprocessor()
    model_builder = ComponentFactory.create_model_builder()

    # Prepare data
    X_train_LSTM, X_test_LSTM = preprocessor.prepare_data(X_train, X_test)

    # Build and compile model
    model = model_builder.build_lstm_model((X_train_LSTM.shape[1], X_train_LSTM.shape[2]))

    # Train the model
    trainer = ComponentFactory.create_trainer(model)
    history = trainer.train(X_train_LSTM, y_train, epochs=50)

    # Evaluate the model
    mse, y_pred = trainer.evaluate(X_test_LSTM, y_test)

    # Plot training and validation loss
    trainer.plot_loss(history)

    # Forecast future rainfall
    forecaster = ComponentFactory.create_forecaster(model)

    # Generate future dates
    last_date = df.index[-1]
    num_days_to_forecast = 30  # Number of days to forecast
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days_to_forecast, freq='D')

    # Define the `update_last_values` function
    def update_last_values(last_values, new_prediction):
        # Update the last values with the new prediction for sequential forecasting
        return np.array([[last_values[0][1], new_prediction]])

    # Prepare last known values for forecasting
    last_values = np.array([[df['minTemp'].iloc[-1], df['Rainfall'].iloc[-1]]])

    # Perform forecasting
    future_predictions = forecaster.forecast(last_values, future_dates, update_last_values)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame(data={'Date': future_dates, 'Forecasted_Rainfall': future_predictions})
    forecast_df.set_index('Date', inplace=True)

    # Visualize forecast
    forecaster.visualize_forecast(df, forecast_df)
