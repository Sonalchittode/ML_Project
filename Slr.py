import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from sklearn.model_selection import train_test_split


data = pd.read_csv("C:\\Users\\sonal\\OneDrive\\Desktop\\DataSet\\study_hours.csv")
x=data[['Study Hours (per Day)']].values
y=data[['Exam Score (out of 100)']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

class Slr:
    def __init__(self):
        self.m = None
        self.d = None

    def fit(self, x_train, y_train):
        num = 0
        den = 0
        for i in range(x_train.shape[0]):
            num += (x_train[i] - x_train.mean()) * (y_train[i] - y_train.mean())
            den += (x_train[i] - x_train.mean()) * (x_train[i] - x_train.mean())
        self.m = num / den
        self.d = y_train.mean() - (self.m * x_train.mean())

    def predict(self, x_test):
        return self.m * x_test + self.d   

slr = Slr()
slr.fit(x_train, y_train)


def show_answer():
    user_input = entry.get()
    try:
        value = float(user_input)   
        prediction = slr.predict(np.array([[value]]))  
        output_label.config(text=f"Predicted Marks: {prediction[0][0]:.2f}")
    except ValueError:
        output_label.config(text="Please enter a valid number!")

root = tk.Tk()
root.geometry("300x150")
root.title("Student Marks Predictor")
root.configure(bg="lightblue")

entry_label = tk.Label(root, text="Enter Time Study:",bg="lightblue", fg="black", font=("Arial", 12))
entry_label.pack(pady=5)   

# Entry
entry = tk.Entry(root, width=20)
entry.pack(pady=5)

# Button
button = tk.Button(root, text="Predict Marks", command=show_answer,bg="green", fg="white", 
                   activebackground="darkgreen", activeforeground="white",
                   font=("Arial", 12, "bold"))
button.pack(pady=5)

# Output Label
output_label = tk.Label(root, text="Result will appear here",bg="white", fg="black", font=("Arial", 12))
output_label.pack(pady=5)

root.mainloop()
