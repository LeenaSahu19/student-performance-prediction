import pandas as pd
import numpy as np

# Reproducible results
np.random.seed(42)

# Number of students
n = 1000

# Generate student data
data = {
    "Student_ID": [f"STU{i:04d}" for i in range(1, n + 1)],
    "Age": np.random.randint(17, 24, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Attendance": np.round(np.random.uniform(50, 100, n), 1),
    "Study_Hours": np.round(np.random.uniform(1, 8, n), 1),
    "Previous_Score": np.round(np.random.uniform(35, 95, n), 1),
    "Assignment_Score": np.round(np.random.uniform(40, 100, n), 1),
    "Internal_Marks": np.round(np.random.uniform(35, 95, n), 1),
    "Participation": np.random.randint(1, 11, n),
    "Previous_Failures": np.random.randint(0, 4, n)
}

df = pd.DataFrame(data)

# Generate Final Score based on academic factors
df["Final_Score"] = (
    0.25 * df["Attendance"]
    + 0.20 * (df["Study_Hours"] / 8 * 100)
    + 0.20 * df["Previous_Score"]
    + 0.15 * df["Assignment_Score"]
    + 0.20 * df["Internal_Marks"]
    + np.random.normal(0, 5, n)
    - df["Previous_Failures"] * 3
)

# Keep scores between 0 and 100
df["Final_Score"] = df["Final_Score"].clip(0, 100).round(1)

# Performance category
def performance_category(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

df["Performance"] = df["Final_Score"].apply(performance_category)

# Risk level
def risk_level(row):
    if row["Final_Score"] < 50 or (
        row["Attendance"] < 65 and row["Study_Hours"] < 3
    ):
        return "High Risk"
    elif row["Final_Score"] < 65 or row["Attendance"] < 75:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df.apply(risk_level, axis=1)

# Save dataset
df.to_csv("data/student_performance.csv", index=False)

print("Dataset created successfully!")
print(f"Total students: {len(df)}")
print("\nFirst 5 records:")
print(df.head())

print("\nPerformance Distribution:")
print(df["Performance"].value_counts())

print("\nRisk Distribution:")
print(df["Risk_Level"].value_counts())