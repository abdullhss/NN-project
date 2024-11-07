import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk





# Read data from csv file 
data = pd.read_csv('./birds.csv')


#  PREPROSSING
# change the gender column and the bird cat -> gender 1 to male , 0 to female ,, A -> 0 B -> , C -> 2 , outlier with IQR
def replace_outliers(df):
    for column in df.select_dtypes(include='number').columns:  # Process only numeric columns
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mean_value = df[column].mean()
        
        df[column] = df[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)

data['gender'] = data['gender'].map({'male': 1, 'female': 0})
data['gender'] = data['gender'].apply(lambda x: np.random.choice([0, 1]) if pd.isna(x) else x)

data['bird category'] = data['bird category'].map({'A': 0 , 'B': 1 , "C" : 2})


replace_outliers(data)

#    /// PREPROSSING ///


print(data.head())

# Separate features (X) and labels (y)
X = data[["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]].values
y = data["bird category"].values


class Perceptron:
    def __init__(self, input_size, learning_rate, epochs, bias):
        self.bias = bias  # Save the bias parameter to the instance
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Initialize weights with an additional slot for the bias if needed
        self.weights = np.zeros(input_size + 1) if self.bias else np.zeros(input_size)

    def predict(self, x):
        if self.bias:
            x = np.insert(x, 0, 1)

        activation = np.dot(x, self.weights)
        return 1 if activation >= 0 else -1  
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = X[i]
                # Add bias term to the input vector if enabled
                if self.bias:
                    x_i = np.insert(x_i, 0, 1)
                
                # Prediction and error calculation
                prediction = self.predict(x_i)
                error = y[i] - prediction

                # Update weights
                self.weights += self.learning_rate * error * x_i

    def evaluate(self, X, y):
        correct_predictions = sum(self.predict(np.insert(x, 0, 1) if self.bias else x) == y_i for x, y_i in zip(X, y))
        return correct_predictions / len(y)





class Adaline:
    def __init__(self, input_size, learning_rate, epochs, mse_threshold, bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.bias = bias
        # Initialize weights: if bias is included, add 1 to the input size
        self.weights = np.zeros(input_size + 1) if bias else np.zeros(input_size)
        
    def predict(self, X):
        # If bias is included, add a column of 1's to X (for the bias term)
        if self.bias:
            X = np.insert(X, 0, 1, axis=1)  # Insert a column of 1's at the beginning
        return np.dot(X, self.weights)  # Perform dot product
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            # Make predictions for the current data
            predictions = self.predict(X)
            # Calculate the errors
            errors = y - predictions
            # Calculate the Mean Squared Error (MSE)
            mse = np.mean(errors**2)
            
            # Check if MSE is less than the threshold, stop training if it is
            if mse < self.mse_threshold:
                print(f"Training stopped early at epoch {epoch + 1} due to MSE threshold")
                break
            
            # Update the weights using the gradient descent rule
            if self.bias:
                X = np.insert(X, 0, 1, axis=1)  # Add bias term (1's) again for weight update
            # Update weights based on error, learning rate, and input data
            self.weights += self.learning_rate * np.dot(X.T, errors) / len(y)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        # Round predictions to nearest integer (binary 0 or 1 classification)
        predictions = np.round(predictions)  # Convert predictions to 0 or 1
        accuracy = np.mean(predictions == y)  # Calculate the accuracy
        return accuracy



#  GUI 

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Neural Network Configuration")
root.geometry("400x550")

# Dropdown 1: Class Choice (Class A, Class B, Class C)
label_class1 = tk.Label(root, text="Select Class 1:")
label_class1.pack(pady=5)
class1_var = tk.StringVar()
dropdown_class1 = ttk.Combobox(root, textvariable=class1_var)
dropdown_class1['values'] = ("Class A", "Class B", "Class C")
dropdown_class1.pack()

# Dropdown 2: Class Choice (Class A, Class B, Class C)
label_class2 = tk.Label(root, text="Select Class 2:")
label_class2.pack(pady=5)
class2_var = tk.StringVar()
dropdown_class2 = ttk.Combobox(root, textvariable=class2_var)
dropdown_class2['values'] = ("Class A", "Class B", "Class C")
dropdown_class2.pack()

# Dropdown 3: Feature Choice (Gender, Body Mass, etc.)
label_feature1 = tk.Label(root, text="Select Feature 1:")
label_feature1.pack(pady=5)
feature1_var = tk.StringVar()
dropdown_feature1 = ttk.Combobox(root, textvariable=feature1_var)

dropdown_feature1['values'] = ("gender", "body_mass", "beak_length", "beak_depth", "fin_length")
dropdown_feature1.pack()

# Dropdown 4: Feature Choice (Gender, Body Mass, etc.)
label_feature2 = tk.Label(root, text="Select Feature 2:")
label_feature2.pack(pady=5)
feature2_var = tk.StringVar()
dropdown_feature2 = ttk.Combobox(root, textvariable=feature2_var)
dropdown_feature2['values'] = ("gender", "body_mass", "beak_length", "beak_depth", "fin_length")
dropdown_feature2.pack()

# Text Field: Learning Rate
label_learning_rate = tk.Label(root, text="Learning Rate:")
label_learning_rate.pack(pady=5)
learning_rate_var = tk.StringVar()
entry_learning_rate = tk.Entry(root, textvariable=learning_rate_var)
entry_learning_rate.pack()

# Text Field: Number of Epochs
label_epochs = tk.Label(root, text="Number of Epochs:")
label_epochs.pack(pady=5)
epochs_var = tk.StringVar()
entry_epochs = tk.Entry(root, textvariable=epochs_var)
entry_epochs.pack()

# Text Field: MSE Threshold
label_mse = tk.Label(root, text="MSE Threshold:")
label_mse.pack(pady=5)
mse_var = tk.StringVar()
entry_mse = tk.Entry(root, textvariable=mse_var)
entry_mse.pack()

# Checkbox: Include Bias
bias_var = tk.BooleanVar()
checkbox_bias = tk.Checkbutton(root, text="Include Bias", variable=bias_var)
checkbox_bias.pack(pady=5)

# Radio Buttons: Algorithm Choice (Perceptron, Adaline)
label_algorithm = tk.Label(root, text="Choose Algorithm:")
label_algorithm.pack(pady=5)
algorithm_var = tk.StringVar(value=0)
radio_perceptron = tk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value=1)
radio_adaline = tk.Radiobutton(root, text="Adaline", variable=algorithm_var, value=0)
radio_perceptron.pack()
radio_adaline.pack()

# Function to capture and print the selected configurations




def submit():
    # try:
        learning_rate = float(learning_rate_var.get())
        epochs = int(epochs_var.get())
        mse_threshold = float(mse_var.get())
        include_bias = bias_var.get()
        algorithm = algorithm_var.get()
        
         # Get selected features from the GUI dropdowns

        feature1 = feature1_var.get()
        feature2 = feature2_var.get()


        class1 = class1_var.get()
        class2 = class2_var.get()

        # Map class names to class numbers
        class_map = {'Class A': 0, 'Class B': 1, 'Class C': 2}
        class1 = class_map.get(class1, None)
        class2 = class_map.get(class2, None)


        # Filter the data based on the class selection and select first 50 rows for each class
        class1_data = data[data['bird category'] == class1].iloc[:50]
        class2_data = data[data['bird category'] == class2].iloc[:50]

        # Split each class data into 30 rows for training and the remaining 20 for testing
        train_class1 = class1_data.iloc[:30]
        test_class1 = class1_data.iloc[30:]
        train_class2 = class2_data.iloc[:30]
        test_class2 = class2_data.iloc[30:]

        # Combine training data from both classes
        X_train = pd.concat([train_class1, train_class2])[[feature1, feature2]].values
        y_train = pd.concat([train_class1, train_class2])["bird category"].values

        # Combine testing data from both classes
        X_test = pd.concat([test_class1, test_class2])[[feature1, feature2]].values
        y_test = pd.concat([test_class1, test_class2])["bird category"].values


        # Display values for verification
        print("Class 1:", class1_var.get())
        print("Class 2:", class2_var.get())
        print("Feature 1:", feature1_var.get())
        print("Feature 2:", feature2_var.get())
        print("Learning Rate:", learning_rate)
        print("Number of Epochs:", epochs)
        print("MSE Threshold:", mse_threshold)
        print("Include Bias:", include_bias)
        print("Algorithm:", "Perceptron" if algorithm == '1' else "Adaline")

        if algorithm == '1': 
            # Initialize and train perceptron
            perceptron = Perceptron(input_size=X_train.shape[1], learning_rate=learning_rate, epochs=epochs, bias=include_bias)
            perceptron.train(X_train, y_train)
            
            # Evaluate the model
            accuracy = perceptron.evaluate(X_test, y_test)
            print("Model accuracy on test set:", accuracy)

            # Display predictions for each test input
            for i, x in enumerate(X_test):
                print(f"Input: {x}, Prediction: {perceptron.predict(x)}, Actual: {y_test[i]}")
       
        elif algorithm == '0':  # Adaline
            adaline = Adaline(input_size=X_train.shape[1], learning_rate=learning_rate, epochs=epochs, mse_threshold=mse_threshold, bias=include_bias)
            adaline.train(X_train, y_train)
            accuracy = adaline.evaluate(X_test, y_test)
            print("Model accuracy on test set:", accuracy)

    # except ValueError:
    #     print("Please enter valid numeric values for Learning Rate, Epochs, and MSE Threshold")


btn_submit = tk.Button(root, text="Submit", command=submit)
btn_submit.pack(pady=20)


root.mainloop()


#  /// GUI ///

