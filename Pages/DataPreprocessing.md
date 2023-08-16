# 1.2 Data Preprocessing
So far, we have been working with synthetic data that arrived in ready-made tensors. However, to apply deep learning in the wild we must extract messy data stored in arbitrary formats, and preprocess it to suit our needs. Fortunately, the pandas [library](https://pandas.pydata.org/) can do much of the heavy lifting. This section, while no substitute for a proper pandas [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html), will give you a crash course on some of the most common routines.

## Reading the Dataset
Comma-separated values (CSV) files are ubiquitous for the storing of tabular (spreadsheet-like) data. In them, each line corresponds to one record and consists of several (comma-separated) fields, e.g., “Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics”. To demonstrate how to load CSV files with pandas, we create a CSV file below ../data/house_tiny.csv. This file represents a dataset of homes, where each row corresponds to a distinct home and the columns correspond to the number of rooms (NumRooms), the roof type (RoofType), and the price (Price).

```Python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```
import pandas and load the dataset with read_csv.
```Python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/8d4ffe1c-f2c9-44bc-a5e3-a37ff02f6c94)
## Data Preparation
In supervised learning, we train models to predict a designated target value, given some set of input values. Our first step in processing the dataset is to separate out columns corresponding to input versus target values. We can select columns either by name or via integer-location based indexing (iloc).

You might have noticed that pandas replaced all CSV entries with value NA with a special NaN (not a number) value. This can also happen whenever an entry is empty, e.g., “3,,,270000”. These are called missing values and they are the “bed bugs” of data science, a persistent menace that you will confront throughout your career. Depending upon the context, missing values might be handled either via imputation or deletion. Imputation replaces missing values with estimates of their values while deletion simply discards either those rows or those columns that contain missing values.

Here are some common imputation heuristics. For categorical input fields, we can treat NaN as a category. Since the RoofType column takes values Slate and NaN, pandas can convert this column into two columns RoofType_Slate and RoofType_nan. A row whose roof type is Slate will set values of RoofType_Slate and RoofType_nan to 1 and 0, respectively. The converse holds for a row with a missing RoofType value.
```Python
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/3cd68a2d-b761-4793-8371-ba5f77d337a5)
For missing numerical values, one common heuristic is to replace the NaN entries with the mean value of the corresponding column.
```Python
inputs = inputs.fillna(inputs.mean())
print(inputs)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/a6fc0f0c-a731-4e5b-ae08-e34b213425e8)
## Conversion to the Tensor Format
```Python
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/383e7250-4b40-4fc6-b2f1-7d54f703d248)

