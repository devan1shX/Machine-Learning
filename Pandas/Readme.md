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
ali_df = df[df['Name'].str.contains('ali', case=False)]


The Query Method

A cleaner syntax for filtering, similar to SQL.

# Simple numeric check
df_query = df.query('Age > 30')

# Using strings
df_query = df.query('City == "Paris"')

# Referencing external variables with @
min_salary = 80000
df_query = df.query('Salary >= @min_salary')

# Complex logic
df_query = df.query('Age < 40 and City != "London"')


🧹 6. Data Cleaning & Preparation

Real-world data is messy.

Handling Missing Data

# Check for nulls
df.isnull().sum()   # Count nulls per column
df.notnull()        # Opposite of isnull

# 1. Dropping Nulls
df.dropna()         # Drop rows if ANY value is null
df.dropna(how='all') # Drop row only if ALL values are null
df.dropna(axis=1)   # Drop COLUMNS with nulls (use carefully)
df.dropna(subset=['Age', 'Salary']) # Drop row if specific cols are null

# 2. Filling Nulls (Imputation)
df.fillna(0)        # Fill all nulls with 0

# Fill with column mean/median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Forward Fill (Propagate last valid observation forward)
# Useful for time series
df.fillna(method='ffill')

# Backward Fill
df.fillna(method='bfill')


Handling Duplicates

# Check for duplicates
df.duplicated().sum()

# View duplicate rows
dupes = df[df.duplicated()]

# Drop duplicates
# keep='first' (default), 'last', or False (drop all copies)
df.drop_duplicates(keep='first', inplace=True)

# Drop duplicates based on specific columns
df.drop_duplicates(subset=['Name', 'ID'], keep='last')


String Manipulation

Accessed via the .str accessor. Vectorized and fast.

# Lowercase/Uppercase
df['City'] = df['City'].str.lower()

# Strip whitespace
df['Name'] = df['Name'].str.strip()

# Split strings into lists
# "John Doe" -> ["John", "Doe"]
df['Name_Split'] = df['Name'].str.split(' ')

# Access split elements (expand=True creates new DataFrame columns)
df[['First', 'Last']] = df['Name'].str.split(' ', expand=True)

# Replace substrings (Regex supported)
df['Phone'] = df['Phone'].str.replace('-', '', regex=False)

# Extract patterns using Regex
df['Email_Domain'] = df['Email'].str.extract(r'@(.+)')


Data Type Conversion

# Convert to integer
df['Age'] = df['Age'].astype(int)

# Convert to float
df['Salary'] = df['Salary'].astype(float)

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert to categorical
# Saves massive amounts of memory for string columns with few unique values
df['Department'] = df['Department'].astype('category')

# Convert numeric to string
df['ID'] = df['ID'].astype(str)


✏️ 7. Transformation & Manipulation

Apply, Map, and ApplyMap

map: Applies a function or dictionary element-wise to a Series.

apply: Applies a function along an axis of a DataFrame or Series.

applymap: Applies a function element-wise to a DataFrame.

# --- Mapping ---
# Replace values using a dictionary
gender_map = {'M': 'Male', 'F': 'Female'}
df['Gender_Full'] = df['Gender'].map(gender_map)

# --- Apply on Series ---
# Create a new column based on existing data
def categorize_salary(salary):
    if salary > 100000: return 'High'
    else: return 'Standard'

df['Salary_Category'] = df['Salary'].apply(categorize_salary)

# --- Apply on DataFrame ---
# axis=1 passes the ROW as a series to the function
df['Bonus'] = df.apply(lambda row: row['Salary'] * 0.10 if row['Age'] > 30 else 0, axis=1)

# --- Vectorized Operations (Faster than Apply) ---
# Always prefer this over apply for math
df['Total_Comp'] = df['Salary'] + df['Bonus']


Sorting & Ranking

# Sort by values
# Ascending
df.sort_values(by='Salary')

# Descending
df.sort_values(by='Salary', ascending=False)

# Multiple columns (Sort by City then Age)
df.sort_values(by=['City', 'Age'], ascending=[True, False])

# Sort by index
df.sort_index()

# Ranking
# Assigns a rank from 1 to N
df['Rank'] = df['Salary'].rank(method='dense', ascending=False)


Binning Data

Converting continuous variables into categorical buckets.

# Pandas cut (Equal width bins)
# Create bins: 0-18, 18-35, 35-60, 60-100
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Pandas qcut (Equal frequency bins / Quantiles)
# Splits data so each bin has the same number of people
df['Salary_Quartile'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])


⚙️ 8. Aggregation & Grouping

The "Split-Apply-Combine" strategy.

GroupBy Mechanics

# 1. Group by a single column
g = df.groupby('Department')

# 2. Aggregate specific columns
# Calculate mean salary by department
avg_sal = g['Salary'].mean()

# 3. Multiple Aggregations
# Get count, mean, and max salary for each department
agg_df = g['Salary'].agg(['count', 'mean', 'max'])

# 4. Group by multiple columns
# Average salary per Dept AND Gender
df.groupby(['Department', 'Gender'])['Salary'].mean()

# 5. Named Aggregation (New Syntax)
# Allows custom naming of result columns
summary = df.groupby('City').agg(
    avg_age=('Age', 'mean'),
    max_salary=('Salary', 'max'),
    total_employees=('Name', 'count')
)


Pivot Tables & Crosstabs

Pivot tables reshape data for summary analysis (like Excel).

# Basic Pivot
# Index: City, Columns: Department, Values: Salary (Mean)
pivot = df.pivot_table(
    values='Salary',
    index='City',
    columns='Department',
    aggfunc='mean',
    fill_value=0
)

# Crosstab
# Frequency table (Counts)
pd.crosstab(df['City'], df['Department'])

# Crosstab with values
pd.crosstab(
    df['City'], 
    df['Department'], 
    values=df['Salary'], 
    aggfunc='mean'
)


🔄 9. Merging & Joining

Combining multiple DataFrames.

Concatenation

Stacking DataFrames on top of each other (vertical) or side-by-side (horizontal).

df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'A': [3, 4]})

# Vertical Stack (Axis 0)
result = pd.concat([df1, df2], ignore_index=True)

# Horizontal Stack (Axis 1)
# Glues columns together
result = pd.concat([df1, df2], axis=1)


Merging (Joins)

Merging on specific keys (like SQL JOIN).

Inner: Intersection of keys (default).

Outer: Union of keys.

Left: Keys from left DF.

Right: Keys from right DF.

users = pd.DataFrame({'key': ['A', 'B', 'C'], 'user': ['user1', 'user2', 'user3']})
logs = pd.DataFrame({'key': ['A', 'B', 'D'], 'log': ['log1', 'log2', 'log3']})

# Inner Join (Only A and B exist in both)
pd.merge(users, logs, on='key', how='inner')

# Left Join (Keep all users, fill missing logs with NaN)
pd.merge(users, logs, on='key', how='left')

# Outer Join (Keep everything)
pd.merge(users, logs, on='key', how='outer')

# Merging on different column names
pd.merge(users, logs, left_on='key', right_on='key_id')


📐 10. Reshaping & MultiIndex

Stack & Unstack

Moving data between index and columns.

Stack: Moves column labels to row index (Makes data "taller").

Unstack: Moves row index to column labels (Makes data "wider").

# Assuming a MultiIndex DataFrame (City, Year)
stacked = df.stack()
unstacked = stacked.unstack()


Melt

Unpivoting data (Wide to Long format).

# Wide format: Columns are 'Day1', 'Day2', 'Day3'
# We want: Column 'Day', Column 'Value'
df_melted = pd.melt(
    df,
    id_vars=['ID', 'Name'],    # Identifiers to keep
    var_name='Day',            # Name for new variable column
    value_name='Score'         # Name for new value column
)


⏳ 11. Time Series Analysis

Pandas was originally built for financial time series.

Date Ranges & Parsing

# Create a range of dates
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

# Set datetime index
df['Date'] = pd.to_datetime(df['Date_String'])
df.set_index('Date', inplace=True)

# Partial String Indexing
# Select all data from January 2023
jan_data = df['2023-01']

# Select specific range
q1_data = df['2023-01':'2023-03']


Resampling

Changing the frequency of your time series (e.g., Daily to Monthly).

# Downsampling (High freq -> Low freq)
# Calculate monthly average
monthly_avg = df.resample('M')['Sales'].mean()

# Upsampling (Low freq -> High freq)
# Fill missing values
daily_data = df.resample('D').asfreq().fillna(method='ffill')


Rolling Windows

Moving averages and window functions.

# 7-day Moving Average
df['7d_MA'] = df['Sales'].rolling(window=7).mean()

# Expanding Window (Cumulative)
df['Cumulative_Sales'] = df['Sales'].expanding().sum()

# Shift (Lag/Lead)
df['Previous_Day_Sales'] = df['Sales'].shift(1)
df['Change'] = df['Sales'] - df['Previous_Day_Sales']


📈 12. Visualization

Pandas integrates deeply with Matplotlib.

import matplotlib.pyplot as plt

# Line Plot (Time series)
df.plot(y='Sales', figsize=(10, 5), title='Daily Sales')

# Bar Plot
df['City'].value_counts().plot(kind='bar')

# Histogram
df['Age'].plot(kind='hist', bins=20)

# Scatter Plot
df.plot(kind='scatter', x='Age', y='Salary')

# Box Plot
df.boxplot(column='Salary', by='Department')

plt.show()


🚀 13. Performance & Best Practices

How to handle large datasets efficiently.

Avoid Loops: Never loop over a DataFrame (for i in range(len(df))). It is extremely slow. Use vectorization or apply.

Use Categorical Types: For string columns with low cardinality (few unique values), convert to category to save memory and speed up groupby.

df['Status'] = df['Status'].astype('category')


Use inplace Sparingly: Most operations return a copy. inplace=True is being deprecated in some future versions of Pandas for certain operations. It often prevents method chaining.

Load Only What You Need: Use usecols in read_csv to load specific columns.

Iterating (If you MUST): Use itertuples() (fastest) instead of iterrows() (slow).

💡 Pro Tip: Method Chaining

Write clean, readable pipelines using parentheses ().

# Clean Pipeline Example
df_clean = (
    pd.read_csv('data.csv')
    .dropna(subset=['ID'])
    .assign(
        Date=lambda x: pd.to_datetime(x['Date']),
        Total=lambda x: x['Price'] * x['Quantity']
    )
    .loc[lambda x: x['Total'] > 100]
    .groupby('Category')
    .agg({'Total': 'sum'})
    .sort_values('Total', ascending=False)
)


<div align="center">
<b>Created with ❤️ for Data Scientists</b>
</div>
