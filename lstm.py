import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


class RainfallPredictor:
    def __init__(self, file_path, scale_type='standard', test_size=0.2, random_state=42):
        """
        Initialize the predictor with dataset file path, scaling type, and test/train split settings.
        """
        self.file_path = file_path
        self.scale_type = scale_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_temp = None
        self.scaler_rainfall = None
        self.model = None
        self.history = None

    def load_and_preprocess_data(self):
        """
        Load the dataset, scale the features, and create lagged features.
        """
        df = pd.read_csv(self.file_path)

        # Select appropriate scaler
        self.scaler_temp = StandardScaler() if self.scale_type == 'standard' else MinMaxScaler()
        self.scaler_rainfall = StandardScaler() if self.scale_type == 'standard' else MinMaxScaler()

        # Scale features
        df['minTemp'] = self.scaler_temp.fit_transform(df[['minTemp']])
        df['Rainfall'] = self.scaler_rainfall.fit_transform(df[['Rainfall']])

        # Create lagged features
        df['minTemp_lag_1'] = df['minTemp'].shift(1)
        df['RainFall_lag_1'] = df['Rainfall'].shift(1)
        df.dropna(inplace=True)

        # Define features (X) and target (y)
        X = df[['minTemp_lag_1', 'RainFall_lag_1']].values
        y = df['Rainfall'].values

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape, dropout_rate=0.2, lstm_units=50):
        """
        Build and compile the LSTM model.
        """
        self.model = Sequential()
        self.model.add(LSTM(lstm_units, activation='relu', input_shape=input_shape)) # why relu?
        self.model.add(Dropout(dropout_rate))  # Add dropout to prevent overfitting # why?
        self.model.add(Dense(1))  # Output layer
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Train the LSTM model on the given training data.
        """
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        """
        Make predictions and evaluate the model using various metrics.
        """
        # Predict
        y_pred = self.model.predict(X_test)

        # Rescale predictions and actual values
        y_test_rescaled = self.scaler_rainfall.inverse_transform(y_test.reshape(-1, 1))
        y_pred_rescaled = self.scaler_rainfall.inverse_transform(y_pred)

        # Metrics
        mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)

        print(f"Mean Squared Error (Rescaled): {mse}")
        print(f"Mean Absolute Error (Rescaled): {mae}")
        print(f"R-Squared: {r2}")

        return y_test_rescaled, y_pred_rescaled

    def plot_training_loss(self):
        """
        Plot training loss over epochs.
        """
        if self.history:
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Model Training Loss')
            plt.legend()
            plt.show()
        else:
            print("No training history available to plot.")

    def plot_predictions(self, y_test_rescaled, y_pred_rescaled):
        """
        Plot actual vs. predicted values.
        """
        plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5)
        plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],
                 [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Rainfall')
        plt.ylabel('Predicted Rainfall')
        plt.title('Actual vs. Predicted Rainfall')
        plt.show()


# Usage Example
if __name__ == "__main__":
    # Initialize the predictor
    predictor = RainfallPredictor(file_path='dataset/raw_dataset/merged_data_timeserie.csv')

    # Preprocess the data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data()

    # Build the model
    predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train the model
    predictor.train(X_train, y_train, epochs=20)

    # Evaluate the model
    y_test_rescaled, y_pred_rescaled = predictor.evaluate(X_test, y_test)

    # Visualize results
    predictor.plot_training_loss()
    predictor.plot_predictions(y_test_rescaled, y_pred_rescaled)
