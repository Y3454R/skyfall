
# LSTM Rainfall Forecasting

This project builds an LSTM-based machine learning pipeline to predict rainfall using historical data. The pipeline includes data preprocessing, training, evaluation, and forecasting.

## Features

- **Data Loading and Preprocessing**: Handles missing values, scaling, and lag feature creation.
- **Model Building**: Constructs a sequential LSTM model using TensorFlow/Keras.
- **Training and Evaluation**: Supports training with validation and computes evaluation metrics like Mean Squared Error (MSE).
- **Forecasting**: Predicts future rainfall values for a specified time period.
- **Visualization**: Plots training/validation loss and historical vs forecasted rainfall.

---

## Project Structure

- `ComponentFactory`: Factory class to create different components (DataLoader, ModelBuilder, etc.).
- `DataLoader`: Loads and preprocesses the dataset.
- `DataPreprocessor`: Reshapes data for LSTM input.
- `ModelBuilder`: Builds and compiles the LSTM model.
- `Trainer`: Handles training, evaluation, and visualization of training loss.
- `Forecaster`: Predicts future rainfall and visualizes the results.

---

## Usage

### **Step 1: Prepare Data**
Provide the dataset in `.csv` format. The data must have at least:
- `Date`: The date column (datetime format).
- `minTemp`: Minimum temperature values.
- `Rainfall`: Rainfall values.

Example file structure:
```csv
Date,minTemp,Rainfall
1970-01-01,15.0,0.0
1970-01-02,16.2,1.5
```

---

### **Step 2: Run the Script**

1. Update the `file_path` variable in the script with your dataset file path.
2. Run the script to:
   - Train the LSTM model.
   - Evaluate it on a test dataset.
   - Forecast future rainfall.

---

### **Output**

#### **Plots**
1. **Training and Validation Loss**:
   - Visualizes model learning over epochs.

2. **Historical vs Forecasted Rainfall**:
   - Compares actual rainfall with predicted values.

#### **Metrics**
- `Mean Squared Error`: A quantitative measure of model performance.

---

### **Dependencies**

- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

### **Results**

1. **Mean Squared Error (MSE)**:
   - The test set MSE indicates the accuracy of rainfall predictions (lower is better).

2. **Forecast Accuracy**:
   - Forecast trends are visualized for the next 30 days.

---

### **Example Results**

1. **Training Loss**:
   ![Training Loss](lstm_training_and_validation_loss.png)

2. **Forecasted Rainfall**:
   ![Forecasted Rainfall](lstm_historical_vs_forecasted_rainfall.png)

---

### **Future Improvements**

- Use additional features (humidity, pressure, etc.) to improve model accuracy.
- Optimize LSTM hyperparameters (e.g., units, learning rate).
- Explore other models like GRU or Transformers for time series forecasting.

---

### **Acknowledgments**

This project utilizes TensorFlow and Keras for building deep learning models.
```

