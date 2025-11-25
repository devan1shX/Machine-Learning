# 🐼 Pandas for Exploratory Data Analysis (EDA)

> A comprehensive guide to data manipulation and analysis with Pandas

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Core Data Structures](#core-data-structures)
- [Creating DataFrames](#creating-dataframes)
- [Reading & Writing Data](#reading--writing-data)
- [DataFrame Inspection](#dataframe-inspection)
- [Indexing & Selection](#indexing--selection)
- [Data Cleaning](#data-cleaning)
- [Statistical Operations](#statistical-operations)
- [Working with Excel](#working-with-excel)

---

## Introduction

**Pandas** is a powerful Python library for data manipulation and analysis, built on NumPy. It provides high-performance data structures optimized for structured data operations.

### Why Pandas?

| Feature | Pandas | Excel | Python Lists |
|---------|--------|-------|--------------|
| **Performance** | ✅ Fast (C-based) | ❌ Slow for large data | ❌ Slower |
| **Data Size** | ✅ Millions of rows | ❌ ~1M row limit | ✅ Large data |
| **Automation** | ✅ Fully scriptable | ❌ Manual operations | ✅ Scriptable |
| **Reproducibility** | ✅ Version controlled | ❌ Hard to track | ✅ Version controlled |
| **Operations** | ✅ Vectorized | ❌ Cell-by-cell | ❌ Manual loops |
| **Missing Data** | ✅ Built-in NaN support | ❌ Manual handling | ❌ Manual handling |

### Key Capabilities
- **Data Cleaning** → Missing values, duplicates, transformations
- **Data Analysis** → Statistical operations, grouping, aggregations
- **Data I/O** → CSV, Excel, SQL, JSON, Parquet, and more
- **Performance** → 10-100x faster than pure Python loops

---

## Installation

### Jupyter Notebook
```python
!pip install pandas --upgrade
```

### Command Line
```bash
# Windows/Linux/Mac
pip install pandas

# With specific version
pip install pandas==2.1.0

# Verify installation
python -c "import pandas as pd; print(pd.__version__)"
```

### Import Convention
```python
import pandas as pd
import numpy as np
```

---

## Core Data Structures

### Series (1D Array with Labels)

A Series is a one-dimensional labeled array holding any data type.

```python
# From list (default index 0, 1, 2...)
scores = pd.Series([92, 85, 78, 95])

# With custom index
scores = pd.Series([92, 85, 78, 95], index=['Anish', 'Manish', 'Rohan', 'Priya'])

# From dictionary (keys become index)
scores = pd.Series({'Anish': 92, 'Manish': 85, 'Rohan': 78, 'Priya': 95})

# Access elements
scores['Anish']  # Returns: 92
scores[0]        # Returns: 92 (if using default index)
```

**Properties:**
- Homogeneous data type
- Labeled index
- Single column structure

### DataFrame (2D Table with Labels)

A DataFrame is a two-dimensional table with labeled rows and columns.

```python
# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Carol'],
    'math': [85, 78, 92],
    'science': [90, 82, 88]
}
df = pd.DataFrame(data)

# From NumPy array
df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])

# From list of dictionaries
data = [
    {'name': 'Alice', 'age': 20},
    {'name': 'Bob', 'age': 22}
]
df = pd.DataFrame(data)
```

**Properties:**
- Multiple columns (each column is a Series)
- Labeled rows and columns
- Heterogeneous data types across columns

---

## Creating DataFrames

### Basic Creation

```python
# Dictionary to DataFrame
student_data = {
    "name": ['Anish', 'Manish', 'Rohan'],
    "marks": [92, 82, 88],
    "city": ['Delhi', 'Mumbai', 'Bangalore']
}
df = pd.DataFrame(student_data)
```

Output:
```
     name  marks       city
0   Anish     92      Delhi
1  Manish     82     Mumbai
2   Rohan     88  Bangalore
```

### With Custom Index

```python
df = pd.DataFrame(student_data, index=['first', 'second', 'third'])
```

Output:
```
          name  marks       city
first    Anish     92      Delhi
second  Manish     82     Mumbai
third    Rohan     88  Bangalore
```

---

## Reading & Writing Data

### CSV Files

```python
# Read CSV
df = pd.read_csv('data.csv')

# Read with specific column as index
df = pd.read_csv('data.csv', index_col=0)

# Write to CSV
df.to_csv('output.csv')                    # With index
df.to_csv('output.csv', index=False)       # Without index
```

### Excel Files

**Required Libraries:**
- `xlrd` - For `.xls` files (Excel 2003)
- `openpyxl` - For `.xlsx` files (Excel 2007+)

```bash
pip install openpyxl xlrd
```

```python
# Read Excel (first sheet by default)
df = pd.read_excel('data.xlsx')

# Read specific sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet2')

# Read all sheets (returns dictionary)
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
df1 = all_sheets['Sheet1']
df2 = all_sheets['Sheet2']

# Write to Excel (single sheet)
df.to_excel('output.xlsx', index=False)

# Write multiple sheets (CORRECT METHOD)
with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Students', index=False)
    df2.to_excel(writer, sheet_name='Grades', index=False)
```

⚠️ **Critical Warning:** `df.to_excel()` overwrites the entire file! Use `ExcelWriter` for multi-sheet files.

### Update Excel Without Losing Other Sheets

```python
# Read and modify
data = pd.read_excel('file.xlsx', sheet_name='Sheet2')
data.loc[0, 'column'] = new_value

# Write back preserving other sheets
with pd.ExcelWriter('file.xlsx', mode='a', engine='openpyxl', 
                    if_sheet_exists='replace') as writer:
    data.to_excel(writer, sheet_name='Sheet2', index=False)
```

### Other Formats

```python
# JSON
df = pd.read_json('data.json')
df.to_json('output.json')

# SQL Database
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
df.to_sql('table_name', conn, if_exists='replace')

# Parquet (big data format)
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet')

# Clipboard (copy from Excel/web, paste to pandas!)
df = pd.read_clipboard()
```

---

## DataFrame Inspection

### Basic Information

```python
# Dimensions (rows, columns)
df.shape  # Returns: (100, 5)

# First N rows
df.head()     # Default: 5 rows
df.head(10)   # First 10 rows

# Last N rows
df.tail()     # Default: 5 rows
df.tail(10)   # Last 10 rows

# Data types and memory
df.info()
```

Example `info()` output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   name        98 non-null     object 
 1   math        95 non-null     float64
 2   science     97 non-null     float64
 3   attendance  100 non-null    int64  
 4   grade       90 non-null     object 
dtypes: float64(2), int64(1), object(2)
memory usage: 4.0+ KB
```

### Statistical Summary

```python
# Numeric columns only
df.describe()
```

Output:
```
          math    science  attendance
count    95.00      97.00      100.00
mean     78.50      82.30       85.20
std      12.40      10.80        8.50
min      45.00      50.00       60.00
25%      70.00      75.00       80.00
50%      80.00      83.00       87.00
75%      88.00      90.00       92.00
max      98.00      99.00       98.00
```

### Index and Columns

```python
# Get index
df.index         # Returns: RangeIndex(start=0, stop=100, step=1)

# Get column names
df.columns       # Returns: Index(['name', 'math', 'science'], dtype='object')

# Rename columns
df.columns = ['A', 'B', 'C']

# Rename specific columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### Transpose

```python
# Swap rows and columns
df.T
```

Before:
```
     Math  Science
Alice  85      90
Bob    78      82
```

After `df.T`:
```
         Alice  Bob
Math        85   78
Science     90   82
```

---

## Indexing & Selection

### Column Selection

```python
# Single column (returns Series)
df['name']
type(df['name'])  # pandas.core.series.Series

# Multiple columns (returns DataFrame)
df[['name', 'math']]
```

### Row Selection with `.loc[]` (Label-based)

```python
# Single row
df.loc[0]                    # Row at index 0

# Multiple rows
df.loc[[0, 2, 5]]           # Rows 0, 2, 5

# Row slice
df.loc[0:5]                 # Rows 0 through 5 (inclusive!)

# Rows and specific columns
df.loc[[0, 1], ['name', 'math']]

# All rows, specific columns
df.loc[:, ['name', 'math']]

# Specific rows, all columns
df.loc[[0, 1], :]
```

### Row Selection with `.iloc[]` (Position-based)

```python
# Single element
df.iloc[0, 0]               # First row, first column

# Multiple rows by position
df.iloc[[0, 3], :]          # Rows at positions 0 and 3

# Row and column slicing
df.iloc[0:5, 2:4]           # First 5 rows, columns 2-3 (exclusive end!)

# Last row
df.iloc[-1, :]

# First column
df.iloc[:, 0]
```

**Key Difference:**
- `.loc[]` → Label-based (inclusive slicing)
- `.iloc[]` → Position-based (exclusive slicing)

### Boolean Indexing

```python
# Single condition
df.loc[df['math'] > 80]

# Multiple conditions (AND)
df.loc[(df['math'] > 80) & (df['science'] > 85)]

# Multiple conditions (OR)
df.loc[(df['math'] < 60) | (df['science'] < 60)]

# NOT condition
df.loc[~(df['name'] == 'Bob')]

# String contains
df.loc[df['name'].str.contains('an')]

# Value in list
df.loc[df['grade'].isin(['A', 'B'])]

# Null values
df.loc[df['math'].isnull()]       # Rows with missing math
df.loc[df['math'].notnull()]      # Rows with non-missing math
```

**Boolean Operators:**
- `&` → AND (both conditions true)
- `|` → OR (either condition true)
- `~` → NOT (negate condition)

⚠️ Always wrap conditions in parentheses when using `&` or `|`

---

## Modifying Values

### Single Cell Update

```python
# CORRECT METHOD (use .loc[])
df.loc[0, 'name'] = 'Alice'

# INCORRECT (chained assignment - causes warnings)
df['name'][0] = 'Alice'  # ❌ Avoid this!
```

### Column Update

```python
# Entire column
df['math'] = 90

# Based on condition
df.loc[df['math'] < 50, 'grade'] = 'F'

# Create new column
df['total'] = df['math'] + df['science']
```

### Copying DataFrames

```python
# Reference (changes affect original) ❌
df_ref = df

# Deep copy (independent) ✅
df_copy = df.copy()

# Alternative deep copy
df_copy = df[:]
```

---

## Data Cleaning

### Detecting Missing Values

```python
# Boolean matrix of missing values
df.isnull()        # True for NaN
df.notnull()       # True for non-NaN

# Count missing values per column
df.isnull().sum()
```

Example output:
```
name        2
math        5
science     3
attendance  0
grade      10
dtype: int64
```

### Creating Missing Values

```python
# Set entire column to None
df['column'] = None

# Set specific values to NaN
import numpy as np
df.loc[0, 'math'] = np.nan
```

### Removing Missing Values

```python
# Drop rows with ANY missing value
df.dropna()

# Drop rows with ALL values missing
df.dropna(how='all')

# Drop columns with ANY missing value
df.dropna(axis=1)

# Drop columns with ALL values missing
df.dropna(axis=1, how='all')

# Drop based on specific columns
df.dropna(subset=['math', 'science'])

# Make changes permanent
df.dropna(inplace=True)
```

**Parameters:**
- `axis=0` → Drop rows (default)
- `axis=1` → Drop columns
- `how='any'` → Drop if ANY value is NaN (default)
- `how='all'` → Drop only if ALL values are NaN

### Removing Duplicates

```python
# Drop duplicate rows (all columns)
df.drop_duplicates()

# Drop based on specific columns
df.drop_duplicates(subset=['name'])

# Keep first occurrence (default)
df.drop_duplicates(keep='first')

# Keep last occurrence
df.drop_duplicates(keep='last')

# Remove all duplicates (keep none)
df.drop_duplicates(keep=False)

# Make changes permanent
df.drop_duplicates(inplace=True)
```

Before:
```
     name  score
0   Alice     85
1     Bob     78
2   Alice     85  ← Duplicate
3   Carol     92
```

After `drop_duplicates()`:
```
     name  score
0   Alice     85
1     Bob     78
3   Carol     92
```

### Dropping Rows/Columns

```python
# Drop rows by index
df.drop([0, 2, 5], axis=0)          # Drop rows 0, 2, 5
df.drop([0, 2, 5])                   # axis=0 is default

# Drop columns by name
df.drop(['name', 'city'], axis=1)    # Drop columns

# Drop single column
df.drop('name', axis=1)

# Make changes permanent
df.drop([0, 2], axis=0, inplace=True)
```

⚠️ `.drop()` returns a copy by default. Use `inplace=True` or reassign: `df = df.drop()`

### Resetting Index

```python
# Reset with default (creates 'index' column)
df.reset_index()

# Reset and drop old index (RECOMMENDED)
df.reset_index(drop=True)

# Make permanent
df.reset_index(drop=True, inplace=True)
```

Before:
```
     name  score
0   Alice     85
5   Carol     88  ← Gap in index
```

After `reset_index(drop=True)`:
```
     name  score
0   Alice     85
1   Carol     88  ← Continuous index
```

### Sorting

```python
# Sort by index (row labels)
df.sort_index()                          # Ascending
df.sort_index(ascending=False)           # Descending

# Sort columns by name
df.sort_index(axis=1)

# Sort by column values
df.sort_values('math')                   # Ascending
df.sort_values('math', ascending=False)  # Descending

# Sort by multiple columns
df.sort_values(['math', 'science'], ascending=[False, True])

# Make permanent
df.sort_values('math', inplace=True)
```

---

## Statistical Operations

### Basic Statistics

```python
# Column-wise operations
df['math'].min()           # Minimum value
df['math'].max()           # Maximum value
df['math'].mean()          # Average
df['math'].median()        # Median (50th percentile)
df['math'].sum()           # Sum
df['math'].std()           # Standard deviation
df['math'].var()           # Variance
df['math'].count()         # Non-null count

# DataFrame-wise (all numeric columns)
df.min()
df.max()
df.mean()
df.sum()
```

### Unique Values

```python
# Get unique values
df['grade'].unique()
# Returns: array(['A', 'B', 'C', 'F'], dtype=object)

# Count unique values
df['grade'].nunique()
# Returns: 4

# Value counts (frequency distribution)
df['grade'].value_counts()
```

Output:
```
A    45
B    30
C    20
F     5
Name: grade, dtype: int64
```

Include NaN in count:
```python
df['grade'].value_counts(dropna=False)
```

### Converting to NumPy

```python
# Convert DataFrame to NumPy array (loses labels)
df.to_numpy()
```

---

## Advanced Operations

### Apply Custom Functions

```python
# Apply to column
df['math'].apply(lambda x: x * 1.1)  # 10% bonus

# Apply to entire DataFrame
df.apply(lambda x: x.max() - x.min())

# Apply row-wise
df.apply(lambda row: row['math'] + row['science'], axis=1)
```

### String Operations

```python
# String methods (for object columns)
df['name'].str.lower()           # Lowercase
df['name'].str.upper()           # Uppercase
df['name'].str.strip()           # Remove whitespace
df['name'].str.replace('a', 'A') # Replace characters
df['name'].str.split(' ')        # Split into list
df['name'].str.len()             # String length
df['name'].str.contains('an')    # Boolean check
```

### Concatenating DataFrames

```python
# Vertical concatenation (stack rows)
pd.concat([df1, df2], axis=0)

# Horizontal concatenation (add columns)
pd.concat([df1, df2], axis=1)

# Ignore original index
pd.concat([df1, df2], ignore_index=True)
```

### Merging DataFrames

```python
# Inner join (only matching keys)
pd.merge(df1, df2, on='student_id', how='inner')

# Left join (all from left, matching from right)
pd.merge(df1, df2, on='student_id', how='left')

# Right join
pd.merge(df1, df2, on='student_id', how='right')

# Outer join (all from both)
pd.merge(df1, df2, on='student_id', how='outer')
```

---

## Common Patterns & Best Practices

### Chaining Operations

```python
# Method chaining for cleaner code
df_clean = (df
    .dropna(subset=['math', 'science'])
    .drop_duplicates()
    .sort_values('math', ascending=False)
    .reset_index(drop=True)
)
```

### Filtering Pipeline

```python
# Complex filtering
high_performers = df.loc[
    (df['math'] > 85) & 
    (df['science'] > 85) & 
    (df['attendance'] > 90)
]
```

### Handling Missing Data Strategy

```python
# Check missing percentage
missing_pct = (df.isnull().sum() / len(df)) * 100

# Drop columns with >50% missing
cols_to_drop = missing_pct[missing_pct > 50].index
df.drop(cols_to_drop, axis=1, inplace=True)

# Fill missing values
df['math'].fillna(df['math'].mean(), inplace=True)  # With mean
df['grade'].fillna('Unknown', inplace=True)         # With constant
```

### Memory Optimization

```python
# Check memory usage
df.info(memory_usage='deep')

# Optimize dtypes
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
```

---

## Quick Reference

### Common Operations

| Operation | Command |
|-----------|---------|
| Read CSV | `pd.read_csv('file.csv')` |
| Write CSV | `df.to_csv('file.csv', index=False)` |
| Read Excel | `pd.read_excel('file.xlsx')` |
| First 5 rows | `df.head()` |
| Last 5 rows | `df.tail()` |
| Shape | `df.shape` |
| Info | `df.info()` |
| Statistics | `df.describe()` |
| Column names | `df.columns` |
| Select column | `df['column']` |
| Select multiple | `df[['col1', 'col2']]` |
| Filter rows | `df[df['col'] > 10]` |
| Missing count | `df.isnull().sum()` |
| Drop missing | `df.dropna()` |
| Drop duplicates | `df.drop_duplicates()` |
| Sort values | `df.sort_values('column')` |
| Reset index | `df.reset_index(drop=True)` |

### Selection Methods

| Method | Type | Use Case |
|--------|------|----------|
| `df['col']` | Column | Select single column |
| `df[['col1', 'col2']]` | Columns | Select multiple columns |
| `df.loc[0]` | Row by label | Label-based indexing |
| `df.iloc[0]` | Row by position | Position-based indexing |
| `df.loc[0:5, 'col']` | Mixed | Row labels + column names |
| `df.iloc[0:5, 2]` | Mixed | Row positions + column position |

### Boolean Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `==` | Equal | `df['col'] == 5` |
| `!=` | Not equal | `df['col'] != 5` |
| `>` | Greater | `df['col'] > 5` |
| `<` | Less | `df['col'] < 5` |
| `>=` | Greater or equal | `df['col'] >= 5` |
| `<=` | Less or equal | `df['col'] <= 5` |
| `&` | AND | `(cond1) & (cond2)` |
| `\|` | OR | `(cond1) \| (cond2)` |
| `~` | NOT | `~(condition)` |

---

## Common Pitfalls

### ⚠️ Chained Assignment
```python
# BAD - causes warnings
df['name'][0] = 'Alice'

# GOOD - use .loc[]
df.loc[0, 'name'] = 'Alice'
```

### ⚠️ Inplace Default
```python
# This doesn't modify df!
df.dropna()

# Do this instead:
df = df.dropna()
# OR
df.dropna(inplace=True)
```

### ⚠️ Index After Deletion
```python
# After dropping rows, reset index
df.drop([1, 3], inplace=True)
df.reset_index(drop=True, inplace=True)
```

### ⚠️ Excel Overwrites
```python
# This deletes all sheets!
df.to_excel('file.xlsx')

# Use ExcelWriter for multi-sheet files
with pd.ExcelWriter('file.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)
```

---

## Additional Resources

- **Official Documentation:** https://pandas.pydata.org/docs/
- **Cheat Sheet:** https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- **10 Minutes to Pandas:** https://pandas.pydata.org/docs/user_guide/10min.html
- **API Reference:** https://pandas.pydata.org/docs/reference/index.html

---

## Summary Workflow

1. **Import & Read Data**
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   ```

2. **Inspect Data**
   ```python
   df.head()
   df.info()
   df.describe()
   df.isnull().sum()
   ```

3. **Clean Data**
   ```python
   df.dropna(inplace=True)
   df.drop_duplicates(inplace=True)
   df.reset_index(drop=True, inplace=True)
   ```

4. **Analyze Data**
   ```python
   df['column'].value_counts()
   df.groupby('category')['value'].mean()
   ```

5. **Export Results**
   ```python
   df.to_csv('cleaned_data.csv', index=False)
   ```

---

**Happy Data Analysis! 🚀**
