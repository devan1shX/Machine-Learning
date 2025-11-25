🐼 The Ultimate Pandas Guide for Data Analysis

Pandas is the cornerstone of data manipulation in Python. It provides high-performance, easy-to-use data structures and data analysis tools. This guide serves as a comprehensive reference for beginners and advanced users alike, covering everything from basic installation to complex time-series analysis and performance optimization.

📑 Table of Contents

📦 Installation & Setup

📊 Core Data Structures

The Series Object

The DataFrame Object

📥 Data Input & Output (I/O)

Reading Data

Writing Data

🔍 Inspection & Exploration

🎯 Selection & Indexing

Column Selection

Row Selection (Loc vs Iloc)

Boolean Indexing & Filtering

The Query Method

🧹 Data Cleaning & Preparation

Handling Missing Data

Handling Duplicates

String Manipulation

Data Type Conversion

✏️ Transformation & Manipulation

Apply, Map, and ApplyMap

Sorting & Ranking

Binning Data

⚙️ Aggregation & Grouping

GroupBy Mechanics

Pivot Tables & Crosstabs

🔄 Merging & Joining

Concatenation

Merging (Joins)

📐 Reshaping & MultiIndex

Stack & Unstack

Melt

⏳ Time Series Analysis

Date Ranges & Parsing

Resampling

Rolling Windows

📈 Visualization

🚀 Performance & Best Practices

📦 1. Installation & Setup

Ensure you have Python installed. Pandas depends on NumPy, which will be installed automatically.

# Standard installation
pip install pandas

# Install optional dependencies for Excel, Parquet, etc.
pip install openpyxl xlrd pyarrow fastparquet


Importing in Python:

import pandas as pd
import numpy as np

# Configuration options (Optional)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.precision', 2)       # Show 2 decimal places


📊 2. Core Data Structures

Pandas has two primary objects: the Series (1D) and the DataFrame (2D).

The Series Object

A Series is a one-dimensional labeled array capable of holding any data type.

# 1. Creating from a List
# Pandas automatically assigns a RangeIndex (0, 1, 2...)
s_list = pd.Series([10, 20, 30, 40, 50])

# 2. Creating from a NumPy Array
# More memory efficient for purely numerical data
arr = np.array([1.5, 2.5, 3.5])
s_arr = pd.Series(arr)

# 3. Creating from a Dictionary
# Keys become the index, values become the data
data_dict = {'Apple': 100, 'Banana': 200, 'Cherry': 150}
s_dict = pd.Series(data_dict)

# 4. Custom Indexing
# Explicitly defining the index labels
s_custom = pd.Series([99, 98, 95], index=['Alice', 'Bob', 'Charlie'])

# Accessing Series Attributes
print(s_custom.values)  # Output: [99 98 95] (NumPy array)
print(s_custom.index)   # Output: Index(['Alice', 'Bob', 'Charlie'], dtype='object')


The DataFrame Object

A DataFrame is a 2-dimensional labeled data structure with columns of potentially different types.

# 1. Creating from a Dictionary of Lists
# Keys become column names
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Paris', 'London', 'Tokyo'],
    'Salary': [70000, 80000, 120000, 95000]
}
df = pd.DataFrame(data)

# 2. Creating from a List of Dictionaries
# Good for row-oriented data
data_rows = [
    {'Name': 'Alice', 'Age': 25},
    {'Name': 'Bob', 'Age': 30}
]
df_rows = pd.DataFrame(data_rows)

# 3. Creating from NumPy Array
# Useful for machine learning datasets (matrices)
matrix = np.random.rand(5, 4)
df_matrix = pd.DataFrame(matrix, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])

# 4. Setting an Index
# You can set one of the columns to be the index (row labels)
df.set_index('Name', inplace=True)


📥 3. Data Input & Output (I/O)

Pandas supports a vast array of file formats.

Reading Data

# --- Text Files ---
# CSV (Comma Separated Values)
df = pd.read_csv('data.csv')

# CSV with specific options
df = pd.read_csv(
    'data.csv',
    sep=',',                 # Delimiter
    header=0,                # Row to use as header (0-based)
    index_col='ID',          # Column to use as index
    usecols=['ID', 'Name'],  # Only load specific columns
    dtype={'ID': int},       # Enforce data types
    parse_dates=['Date'],    # Parse date columns automatically
    na_values=['?', 'N/A']   # Custom missing values
)

# TSV (Tab Separated Values)
df = pd.read_csv('data.tsv', sep='\t')

# --- Excel Files ---
# Requires 'openpyxl'
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read all sheets (returns a dictionary of DataFrames)
dfs = pd.read_excel('data.xlsx', sheet_name=None)

# --- Other Formats ---
# JSON
df = pd.read_json('data.json')

# SQL (Requires SQLAlchemy)
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df = pd.read_sql('SELECT * FROM users', engine)

# Parquet (Fast, compressed binary format)
df = pd.read_parquet('data.parquet')

# HTML (Web scraping tables)
# Returns a list of DataFrames
dfs = pd.read_html('[https://en.wikipedia.org/wiki/Python_(programming_language](https://en.wikipedia.org/wiki/Python_(programming_language))')
df = dfs[1]  # Select the second table on the page

# Clipboard (Copy from Excel/Web and run this)
df = pd.read_clipboard()


Writing Data

# CSV
# index=False prevents writing the row numbers to the file
df.to_csv('output.csv', index=False)

# Excel
# Writing to a specific sheet
df.to_excel('output.xlsx', sheet_name='Summary', index=False)

# Writing multiple DataFrames to one Excel file
with pd.ExcelWriter('multisheet.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# JSON
df.to_json('output.json', orient='records', lines=True)

# Parquet
df.to_parquet('output.parquet')

# SQL
df.to_sql('table_name', con=engine, if_exists='replace', index=False)


🔍 4. Inspection & Exploration

Get to know your data before modifying it.

# --- Basic Views ---
df.head(10)      # First 10 rows
df.tail(5)       # Last 5 rows
df.sample(5)     # Random 5 rows (good for checking bias)

# --- Structural Information ---
df.shape         # Tuple: (rows, columns)
df.columns       # List of column names
df.index         # Information about the index
df.dtypes        # Data type of each column

# --- Detailed Summaries ---
# Info: Non-null counts, data types, memory usage
df.info()

# Describe: Summary statistics for numerical columns
# (count, mean, std, min, 25%, 50%, 75%, max)
df.describe()

# Describe: Summary for categorical columns
# (count, unique, top, freq)
df.describe(include=['object', 'category'])

# --- Unique Values ---
df['City'].unique()       # Array of distinct values
df['City'].nunique()      # Count of distinct values
df['City'].value_counts() # Frequency count of each value (descending)

# Normalizing value counts (percentage)
df['City'].value_counts(normalize=True)


🎯 5. Selection & Indexing

Selecting data is the most common operation. Use .loc and .iloc for robust code.

Column Selection

# Select single column (Returns Series)
s = df['Name']

# Select multiple columns (Returns DataFrame)
# Note the double brackets
subset = df[['Name', 'Age', 'Salary']]

# Dynamic selection using string methods
# Select columns starting with 'Sales_'
cols = [c for c in df.columns if c.startswith('Sales_')]
df_sales = df[cols]


Row Selection (Loc vs Iloc)

Method

Purpose

Syntax

Note

.loc

Label-based

df.loc[row_label, col_label]

Inclusive of the end label.

.iloc

Integer-based

df.iloc[row_idx, col_idx]

Exclusive of the end index (standard Python slicing).

# --- .loc (Label) examples ---
# Select row with index 'Alice' and column 'Age'
val = df.loc['Alice', 'Age']

# Select rows 'Alice' through 'Charlie' and all columns
subset = df.loc['Alice':'Charlie', :]

# Select specific rows and specific columns
subset = df.loc[['Alice', 'David'], ['Age', 'City']]

# --- .iloc (Position) examples ---
# Select row at position 0 and column at position 1
val = df.iloc[0, 1]

# Select first 5 rows and first 3 columns
subset = df.iloc[0:5, 0:3]

# Select the last row
last_row = df.iloc[-1, :]


Boolean Indexing & Filtering

Filtering data based on conditions.

# 1. Create a mask (Boolean Series)
mask = df['Age'] > 30

# 2. Apply mask
over_30 = df[mask]

# Direct filtering
high_earners = df[df['Salary'] > 100000]

# Multiple Conditions
# AND (&), OR (|), NOT (~)
# Parentheses are MANDATORY due to Python operator precedence
complex_filter = df[(df['Age'] > 25) & (df['City'] == 'New York')]

# 'Is In' Check (categorical filtering)
target_cities = ['London', 'Paris', 'Tokyo']
travel_df = df[df['City'].isin(target_cities)]

# String filtering
# Select names containing "ali" (case insensitive)
ali_df = df[df['Name'].str.contains('
