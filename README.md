# Cancer-Detection-And-predicative-analysis
Cancer Detection And predicative analysis using Python  
Cancer remains one of the most challenging health concerns worldwide, with early detection being pivotal for effective treatment and improved outcomes. The project objectives are to enhance cancer detection and predictive analysis through a facts-pushed method using Python programming. Specifically, leveraging libraries which include Beautiful Soup for net scraping and pandas for records management, the device will acquire and prioritize scientific data from various assets. This consists of affected person facts, diagnostic results, and applicable medical literature. The machine will be conscious of two essential areas: evaluating diagnostic techniques for most cancer detection and predicting patient consequences. For cancer detection, the device will gather facts from clinical imaging, genetic markers, and medical reviews. Data preprocessing techniques, consisting of cleaning and integration, will make certain data high-quality and consistent. Advanced engineering strategies including characteristic extraction, classification, and version design might be carried out to decorate the validity and accuracy of the analysis. Machine studying models can be evolved to predict most cancers' presence, subtype, and development based on affected person traits. Predictive analytics will play an important function in figuring out patient effects by studying factors consisting of treatment effectiveness, sickness recurrence, and survival prices. The gadget will make use of predictive modeling to tailor remedy plans and interventions based on character-affected person profiles. Visualization strategies will be employed to give insights and findings in a comprehensible manner, assisting clinical experts in choice-making procedures.
Q=What are the most commonly purchased Cancer causes and their corresponding relevance score?
Purpose: Analyzing the most used  causes and their detection can help identify trends and provided recommendations.
Code:
import pandas as pd
# Load cancer causes detection data
cancer_detection_data = pd.read_csv('/content/detection_Cancer.csv.csv')
# Load Home Depot medicine search relevance data
cancer_data = pd.read_csv('/content/train.csv', encoding='latin1')
# Merge datasets
merged_data = pd.merge(cancer_detection_data, cancer_data, on='medicineid', how='inner')
# Group by medicine and calculate average relevance
avg_relevance_by_medicine = merged_data.groupby('medicineid')['relevance'].mean()

# Get top 10 causes by relevance
top_causes = avg_relevance_by_medicine.sort_values(ascending=False).head(10)
print("Top 10 Cancer Causes by Average Relevance:")
print(top_causes)

Code for Creating Arrays, basic operations (Array Join, split, search,sort):
import numpy as np

# Generate synthetic cancer data (for demonstration purposes)
num_samples = 100
num_features = 10

# Assume each row represents a patient and each column represents a feature
cancer_data = np.random.rand(num_samples, num_features)  # Synthetic cancer data

# Basic array operations

# Joining arrays
joined_array = np.concatenate((cancer_data, np.random.rand(num_samples, num_features)), axis=0)
print("Joined array shape:", joined_array.shape)

# Splitting arrays
split_arrays = np.array_split(joined_array, 2, axis=0)
print("Split arrays shapes:", [arr.shape for arr in split_arrays])

# Searching for specific features
feature_to_search = 0.5
indices = np.where(joined_array[:, 0] > feature_to_search)[0]
print("Indices where feature > 0.5:", indices)

# Sorting arrays
sorted_indices = np.argsort(joined_array[:, 0])
sorted_array = joined_array[sorted_indices]
print("Sorted array based on feature 0:", sorted_array[:5, 0])  # Displaying first 5 values of the sorted feature


Code for Indexing, Slicing and iteDetection, copying arrays:
import numpy as np

# Generate synthetic cancer data (for demonstration purposes)
num_samples = 100
num_features = 10

# Assume each row represents a patient and each column represents a feature
cancer_data = np.random.rand(num_samples, num_features)  # Synthetic cancer data

# Indexing and Slicing
# Accessing specific patient data (row)
patient_5_data = cancer_data[5]
print("Data of patient 5:", patient_5_data)

# Accessing specific feature data (column)
feature_3_data = cancer_data[:, 3]
print("Data of feature 3 across all patients:", feature_3_data)

# Slicing for specific subsets of data
subset_data = cancer_data[10:20, 2:5]  # Rows 10 to 19, columns 2 to 4
print("Subset of data:", subset_data)

# Iteration
# Iterating over rows (patients) and printing the first feature value of each patient
for i, patient_data in enumerate(cancer_data):
    print("Patient", i+1, "Feature 1 value:", patient_data[0])

# Copying arrays
# Creating a copy of the cancer data
cancer_data_copy = np.copy(cancer_data)

# Modify the copied array
cancer_data_copy[0, 0] = 999  # Modify the first patient's first feature in the copied array

# Verify the modification in the copied array
print("\nOriginal data of patient 1:", cancer_data[0, 0])
print("Modified data of patient 1 in copied array:", cancer_data_copy[0, 0])

Code for Search for information using series:
import pandas as pd

# Assume we have a dataset with patient information for cancer detection
# Let's create a sample dataset for demonstration
data = {
    'Patient_ID': [101, 102, 103, 104, 105],
    'Age': [45, 55, 60, 52, 48],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Tumor_Size': [3.5, 4.2, 2.8, 5.1, 3.9],
    'Cancer_Type': ['Breast', 'Prostate', 'Lung', 'Colon', 'Breast']
}

# Create a DataFrame from the sample dataset
df = pd.DataFrame(data)

# Convert the 'Patient_ID' column to the index of the DataFrame
df.set_index('Patient_ID', inplace=True)

# Convert the 'Cancer_Type' column to a Pandas Series
cancer_type_series = df['Cancer_Type']

# Search for patients with a specific cancer type (e.g., Breast cancer)
breast_cancer_patients = cancer_type_series[cancer_type_series == 'Breast']
print("Patients with Breast cancer:")
print(breast_cancer_patients)

# Search for patients with age above 50
patients_above_50 = df[df['Age'] > 50]
print("\nPatients above 50 years old:")
print(patients_above_50)

# Search for patients with tumor size between 3.0 and 4.0
patients_with_tumor_size_between_3_and_4 = df[(df['Tumor_Size'] >= 3.0) & (df['Tumor_Size'] <= 4.0)]
print("\nPatients with tumor size between 3.0 and 4.0:")
print(patients_with_tumor_size_between_3_and_4)

Code for Index hierarchy, index objects, index re-indexing:
import pandas as pd

# Assume we have a dataset with patient information for cancer detection
# Let's create a sample dataset for demonstration
data = {
    'Patient_ID': [101, 102, 103, 104, 105],
    'Age': [45, 55, 60, 52, 48],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Tumor_Size': [3.5, 4.2, 2.8, 5.1, 3.9],
    'Cancer_Type': ['Breast', 'Prostate', 'Lung', 'Colon', 'Breast']
}

# Create a DataFrame from the sample dataset
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Summary statistics of numerical columns
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Information about the DataFrame
print("\nInformation about the DataFrame:")
print(df.info())

# Number of unique values in each column
print("\nNumber of unique values in each column:")
print(df.nunique())

# Value counts of categorical columns
print("\nValue counts of categorical columns:")
print(df['Cancer_Type'].value_counts())

# Grouping and aggregation
# Average tumor size by cancer type
average_tumor_size_by_type = df.groupby('Cancer_Type')['Tumor_Size'].mean()
print("\nAverage tumor size by cancer type:")
print(average_tumor_size_by_type)

