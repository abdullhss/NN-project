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
    def __init__(self, select_eta, select_epoch, select_mse_thre, select_bias):
        self.select_eta = select_eta
        self.select_epoch = select_epoch
        self.select_mse_thre = select_mse_thre
        self.select_bias = select_bias

    def signum(self, x):
        return np.where(x >= 0, 1, 0)

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def train(self, X, y):
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        self.weights = np.random.rand(X.shape[1])
        mse_list = []

        for i in range(self.select_epoch):
            net_value = np.dot(X, self.weights)
            Pred_output=self.signum(net_value)
            error = y - Pred_output

            mse = self.calculate_mse(y, Pred_output)
            mse_list.append(mse)

            self.weights += self.select_eta * np.dot(X.T, error)

            if mse < self.select_mse_thre:
                break

        return mse_list

    def train(self, X):

        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        net_value = np.dot(X, self.weights)
        Pred_output = self.signum(net_value)

        return Pred_output



class Adaline:
    def __init__(self, select_eta, select_epoch, select_mse_thre, select_bias):
        self.select_eta = select_eta
        self.select_epoch = select_epoch
        self.select_mse_thre = select_mse_thre
        self.select_bias = select_bias

    def act_linear(self, x):
        return x

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def fit(self, X, y):
        # Initialize the weights and Add_bias
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        self.weights = np.random.rand(X.shape[1])
        mse_list = []

        for i in range(self.select_epoch):
            net_value = np.dot(X, self.weights)
            predic_output = self.act_linear(net_value)
            error = y - predic_output

            mse = self.calculate_mse(y, predic_output)
            mse_list.append(mse)

            self.weights += self.select_eta * np.dot(X.T, error)

            # if MSE threshold found
            if mse < self.select_mse_thre:
                break

        return mse_list

    def predict(self, X):
        # Add bias term to the features
        if self.select_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.zeros(X.shape[0]), X]

        # Calculate the net value
        net_value = np.dot(X, self.weights)
        net_value = np.where(net_value >= 0.5, 1, 0)

        # Calculate the actual output
        predic_output = self.act_linear(net_value)
        return predic_output


















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
            perceptron = Perceptron(learning_rate , epochs , mse_threshold , include_bias)
            mseList = perceptron.train(X_train , y_train)
            ypred = perceptron.predict(X_test)
            finalMSE = mseList[-1]
            print(f'Final Mean Squared Error for train (MSE): {finalMSE:.6f}')
            accuracy = np.mean(y_test == ypred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

            
        elif algorithm == '0':  # Adaline

            adaline = Adaline(learning_rate , epochs,mse_threshold , include_bias)
            mseList = adaline.train(X_train , y_train)
            ypred = adaline.predict(X_test)
            finalMSE = mseList[-1]
            print(f'Final Mean Squared Error for train (MSE): {finalMSE}')
            accuracy = adaline.calculate_accuracy(y_test, ypred)
            print(f'Accuracy test: {accuracy * 100:.2f}%')


btn_submit = tk.Button(root, text="Submit", command=submit)
btn_submit.pack(pady=20)


root.mainloop()


#  /// GUI ///

