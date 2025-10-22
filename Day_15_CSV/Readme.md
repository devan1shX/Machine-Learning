# 📊 Complete Guide to Working with CSV Files in Python using Pandas

> A comprehensive, beginner-friendly guide to mastering CSV operations for Machine Learning

---

## 📑 Table of Contents

1. [Importing Pandas](#1-importing-pandas)
2. [Opening a Local CSV File](#2-opening-a-local-csv-file)
3. [Opening CSV from URL](#3-opening-csv-from-url)
4. [The `sep` Parameter](#4-the-sep-parameter)
5. [The `names` Parameter](#5-the-names-parameter)
6. [The `index_col` Parameter](#6-the-index_col-parameter)
7. [The `header` Parameter](#7-the-header-parameter)
8. [The `usecols` Parameter](#8-the-usecols-parameter)
9. [The `squeeze` Parameter](#9-the-squeeze-parameter)
10. [The `skiprows` and `nrows` Parameters](#10-the-skiprows-and-nrows-parameters)
11. [The `encoding` Parameter](#11-the-encoding-parameter)
12. [Handling Bad Lines](#12-handling-bad-lines)
13. [The `dtype` Parameter](#13-the-dtype-parameter)
14. [Handling Dates](#14-handling-dates)
15. [The `parse_dates` Parameter](#15-the-parse_dates-parameter)
16. [The `converters` Parameter](#16-the-converters-parameter)
17. [The `na_values` Parameter](#17-the-na_values-parameter)
18. [Loading Large Datasets in Chunks](#18-loading-large-datasets-in-chunks)

---

## 1. Importing Pandas

**What is Pandas?**  
Pandas is like a super-powered Excel for Python! It helps us read, manipulate, and analyze data easily.

```python
import pandas as pd
import numpy as np

# Check your pandas version
print(f"Pandas version: {pd.__version__}")
```

**Output:**
```
Pandas version: 2.0.3
```

**Why use `pd`?**  
We use `pd` as a shortcut so we don't have to type "pandas" every time. It's like giving your friend a nickname!

---

## 2. Opening a Local CSV File

**What is a CSV file?**  
CSV stands for "Comma Separated Values". It's like a simple spreadsheet where data is separated by commas.

```python
# Basic way to read a CSV file
df = pd.read_csv('aug_train.csv')

# Display first 5 rows
print(df.head())

# Display basic information
print(f"\nShape of data: {df.shape}")  # (rows, columns)
print(f"Column names: {df.columns.tolist()}")
```

**Example Output:**
```
   enrollee_id      city  city_development_index  gender  \
0        8949  city_103                   0.920    Male   
1       29725   city_40                   0.776    Male   
2       11561   city_21                   0.624     NaN   
3       33241  city_115                   0.789     NaN   
4         666  city_162                   0.767    Male   

   relevent_experience enrolled_university education_level  \
0    Has relevent experience       no_enrollment        Graduate   
1   No relevent experience       no_enrollment        Graduate   
2   No relevent experience    Full time course        Graduate   
3   No relevent experience                 NaN        Graduate   
4    Has relevent experience       no_enrollment         Masters   

Shape of data: (19158, 14)
Column names: ['enrollee_id', 'city', 'city_development_index', 'gender', ...]
```

**Visual Representation:**
```
CSV File (aug_train.csv):
+-------------+---------+----------+--------+
| enrollee_id | city    | gender   | ...    |
+-------------+---------+----------+--------+
| 8949        | city_103| Male     | ...    |
| 29725       | city_40 | Male     | ...    |
| 11561       | city_21 | NaN      | ...    |
+-------------+---------+----------+--------+
                    ⬇️
              Pandas DataFrame
```

---

## 3. Opening CSV from URL

**Why read from URL?**  
Sometimes datasets are hosted online (like on GitHub). Instead of downloading, we can read them directly!

```python
import requests
from io import StringIO

# Method 1: Direct URL (if the URL directly points to raw CSV)
url = "https://raw.githubusercontent.com/username/repo/main/data.csv"
df = pd.read_csv(url)

# Method 2: Using requests library (more control)
url = "https://raw.githubusercontent.com/username/repo/main/aug_train.csv"
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Convert text content to StringIO object (like a file)
    data = StringIO(response.text)
    df = pd.read_csv(data)
    print("Data loaded successfully!")
    print(df.head())
else:
    print(f"Failed to load data. Status code: {response.status_code}")
```

**Breaking it down:**

1. **`requests.get(url)`** - Asks the website for the file (like asking someone to hand you a book)
2. **`response.status_code`** - Checks if we got it successfully (200 means "Yes!")
3. **`response.text`** - The actual CSV content as text
4. **`StringIO()`** - Converts text into a file-like object that pandas can read

**Visual Flow:**
```
GitHub URL
    ⬇️ (requests.get)
Raw CSV Text
    ⬇️ (StringIO)
File-like Object
    ⬇️ (pd.read_csv)
Pandas DataFrame
```

---

## 4. The `sep` Parameter

**What is `sep`?**  
`sep` stands for "separator". It tells pandas what character separates values in your file.

**Common separators:**
- `,` (comma) - Default, most common
- `;` (semicolon) - Common in European countries
- `\t` (tab) - Tab-separated values (TSV)
- `|` (pipe) - Sometimes used in databases

```python
# CSV with comma separator (default)
df_comma = pd.read_csv('data.csv', sep=',')

# CSV with semicolon separator
df_semicolon = pd.read_csv('data.csv', sep=';')

# Tab-separated file
df_tab = pd.read_csv('data.tsv', sep='\t')

# Pipe-separated file
df_pipe = pd.read_csv('data.txt', sep='|')
```

**Example:**

**File with comma separator:**
```
name,age,city
John,25,NYC
Jane,30,LA
```

**File with semicolon separator:**
```
name;age;city
John;25;NYC
Jane;30;LA
```

```python
# If you use wrong separator
df_wrong = pd.read_csv('semicolon_file.csv', sep=',')
print(df_wrong.head())
# Output: Everything in one column! 😱
#   name;age;city
# 0  John;25;NYC
# 1  Jane;30;LA

# Correct separator
df_correct = pd.read_csv('semicolon_file.csv', sep=';')
print(df_correct.head())
# Output: Properly separated! 😊
#    name  age city
# 0  John   25  NYC
# 1  Jane   30   LA
```

---

## 5. The `names` Parameter

**What is `names`?**  
The `names` parameter lets you provide custom column names, especially useful when:
- Your CSV doesn't have headers
- You want to rename columns while loading

```python
# CSV without headers
# File content:
# 8949,city_103,0.920,Male
# 29725,city_40,0.776,Male

# Provide custom column names
column_names = ['id', 'city', 'dev_index', 'gender']
df = pd.read_csv('data_no_header.csv', 
                 names=column_names,
                 header=None)  # Tell pandas there's no header row

print(df.head())
```

**Output:**
```
      id      city  dev_index gender
0   8949  city_103      0.920   Male
1  29725   city_40      0.776   Male
```

**Before and After:**
```
BEFORE (Raw CSV without names parameter):
     0         1      2     3
0  8949  city_103  0.920  Male
1 29725   city_40  0.776  Male

AFTER (With names parameter):
    id      city  dev_index gender
0  8949  city_103      0.920   Male
1 29725   city_40      0.776   Male
```

**Combining `sep` and `names`:**
```python
# For a pipe-separated file without headers
df = pd.read_csv('data.txt', 
                 sep='|',
                 names=['employee_id', 'name', 'department', 'salary'],
                 header=None)
```

---

## 6. The `index_col` Parameter

**What is an index?**  
The index is like a row number or ID for each row. By default, pandas creates numbers (0, 1, 2...), but you can use a column from your data!

```python
# Without index_col (default behavior)
df = pd.read_csv('aug_train.csv')
print(df.head())
```

**Output:**
```
   enrollee_id      city  gender  ...
0        8949  city_103    Male  ...
1       29725   city_40    Male  ...
2       11561   city_21     NaN  ...
```
*(Notice the 0, 1, 2 on the left? That's the default index)*

```python
# With index_col - Use enrollee_id as index
df = pd.read_csv('aug_train.csv', index_col='enrollee_id')
print(df.head())
```

**Output:**
```
                  city  gender  ...
enrollee_id                     ...
8949          city_103    Male  ...
29725          city_40    Male  ...
11561          city_21     NaN  ...
```
*(Now enrollee_id is the index!)*

**Using column number:**
```python
# Use first column (column 0) as index
df = pd.read_csv('aug_train.csv', index_col=0)

# Use multiple columns as index (MultiIndex)
df = pd.read_csv('sales_data.csv', index_col=['year', 'month'])
```

**When to use `index_col`:**
- ✅ When you have a unique ID column (like student_id, order_id)
- ✅ When you want faster lookups by that column
- ✅ When doing time series analysis with dates

**Visual Comparison:**
```
WITHOUT index_col:
   ID  Name  Score          WITH index_col='ID':
0  101 Alice    95          ID   Name  Score
1  102   Bob    87          101  Alice    95
2  103  Carl    92          102    Bob    87
                             103   Carl    92
```

---

## 7. The `header` Parameter

**What is `header`?**  
The `header` parameter tells pandas which row contains column names.

**Possible values:**
- `header=0` - First row has column names (DEFAULT)
- `header=None` - No header row, pandas creates numbers
- `header=1` - Second row has column names (skip first row)
- `header=[0,1]` - Multi-level column names

```python
# Default: First row is header
df = pd.read_csv('data.csv', header=0)
# Same as: df = pd.read_csv('data.csv')

# No header in file
df = pd.read_csv('data.csv', header=None)
print(df.head())
# Output:
#       0         1      2
# 0  8949  city_103  0.920
# 1 29725   city_40  0.776

# Skip first row, use second row as header
df = pd.read_csv('data.csv', header=1)

# No header + custom names
df = pd.read_csv('data.csv', 
                 header=None,
                 names=['id', 'city', 'index'])
```

**Real Example:**

**File with metadata in first row:**
```
# Dataset created on 2024-01-15
enrollee_id,city,gender
8949,city_103,Male
29725,city_40,Male
```

```python
# Skip the metadata line, use second line as header
df = pd.read_csv('data_with_metadata.csv', header=1)
```

**Multi-level headers:**
```
Year,Year,Score,Score
2023,2024,Math,English
100,110,85,90
```

```python
df = pd.read_csv('multi_header.csv', header=[0,1])
# Creates multi-level columns: (Year, 2023), (Year, 2024), etc.
```

---

## 8. The `usecols` Parameter

**What is `usecols`?**  
When your CSV has many columns but you only need a few, `usecols` lets you load only specific columns. This saves memory and speeds up loading!

```python
# Load only specific columns by name
df = pd.read_csv('aug_train.csv', 
                 usecols=['enrollee_id', 'city', 'gender', 'education_level'])

print(df.head())
```

**Output:**
```
   enrollee_id      city  gender education_level
0        8949  city_103    Male        Graduate
1       29725   city_40    Male        Graduate
2       11561   city_21     NaN        Graduate
```

**Using column numbers:**
```python
# Load first 4 columns (0, 1, 2, 3)
df = pd.read_csv('aug_train.csv', usecols=[0, 1, 2, 3])
```

**Using a function:**
```python
# Load only columns containing 'experience' in name
df = pd.read_csv('aug_train.csv', 
                 usecols=lambda column: 'experience' in column.lower())

# Load only numeric columns
def is_numeric_column(col):
    return col in ['city_development_index', 'training_hours']

df = pd.read_csv('aug_train.csv', usecols=is_numeric_column)
```

**Why use `usecols`?**

**Without usecols:**
```python
df = pd.read_csv('huge_file.csv')  # Loads all 100 columns
# Memory usage: 500 MB 😰
```

**With usecols:**
```python
df = pd.read_csv('huge_file.csv', usecols=['id', 'name', 'score'])
# Memory usage: 50 MB 😊
# 10x faster loading!
```

**Visual Representation:**
```
Original CSV (10 columns):
[ID][Name][Age][City][State][Country][Score][Grade][Phone][Email]
                          ⬇️ usecols=['ID', 'Name', 'Score']
Loaded DataFrame (3 columns):
[ID][Name][Score]
```

---

## 9. The `squeeze` Parameter

**What is `squeeze`?**  
If you're loading only ONE column, `squeeze=True` converts the DataFrame to a Series (simpler structure).

```python
# Without squeeze (returns DataFrame)
df = pd.read_csv('aug_train.csv', usecols=['city'])
print(type(df))  # <class 'pandas.core.frame.DataFrame'>
print(df.head())
```

**Output:**
```
       city
0  city_103
1   city_40
2   city_21
```

```python
# With squeeze (returns Series)
series = pd.read_csv('aug_train.csv', 
                     usecols=['city'], 
                     squeeze=True)
print(type(series))  # <class 'pandas.core.series.Series'>
print(series.head())
```

**Output:**
```
0    city_103
1     city_40
2     city_21
Name: city, dtype: object
```

**When to use `squeeze`?**
- ✅ When you need just one column for analysis
- ✅ When working with mathematical operations on single column
- ✅ When you want simpler code for single column operations

**Practical Example:**
```python
# Get all training hours as a Series
hours = pd.read_csv('aug_train.csv', 
                    usecols=['training_hours'], 
                    squeeze=True)

# Now you can directly use Series operations
print(f"Average training hours: {hours.mean()}")
print(f"Max training hours: {hours.max()}")
```

**Note:** In newer pandas versions (2.0+), `squeeze` is deprecated. Use this instead:
```python
series = pd.read_csv('data.csv', usecols=['city'])['city']
```

---

## 10. The `skiprows` and `nrows` Parameters

### `skiprows` - Skip rows from the beginning

**What is `skiprows`?**  
Skips specified number of rows or specific rows from the start of the file.

```python
# Skip first 3 rows
df = pd.read_csv('aug_train.csv', skiprows=3)

# Skip specific rows (0-indexed)
df = pd.read_csv('aug_train.csv', skiprows=[0, 2, 4])
# Skips 1st, 3rd, and 5th rows

# Skip rows using a function
df = pd.read_csv('aug_train.csv', 
                 skiprows=lambda x: x > 0 and x % 2 == 0)
# Skips all even-numbered rows except header
```

**Example File:**
```
# This is metadata - Line 0
# Created: 2024-01-15 - Line 1
# Version: 1.0 - Line 2
enrollee_id,city,gender - Line 3 (HEADER)
8949,city_103,Male - Line 4
29725,city_40,Male - Line 5
```

```python
# Skip first 3 lines of metadata
df = pd.read_csv('data.csv', skiprows=3)
# Now header is 'enrollee_id,city,gender'
```

### `nrows` - Load only N rows

**What is `nrows`?**  
Loads only the first N rows of data (after header). Super useful for previewing large files!

```python
# Load only first 100 rows
df = pd.read_csv('aug_train.csv', nrows=100)
print(f"Loaded {len(df)} rows")  # Output: Loaded 100 rows

# Load first 5 rows for quick preview
df_preview = pd.read_csv('huge_dataset.csv', nrows=5)
print(df_preview)
```

**Combining `skiprows` and `nrows`:**
```python
# Skip first 1000 rows, then load next 500 rows
df = pd.read_csv('huge_file.csv', skiprows=range(1, 1001), nrows=500)
# This loads rows 1001-1500
```

**Practical Use Case - Sampling Large Dataset:**
```python
# Get first 1000 rows
sample_1 = pd.read_csv('big_data.csv', nrows=1000)

# Get middle 1000 rows (skip 10000, load 1000)
sample_2 = pd.read_csv('big_data.csv', 
                       skiprows=range(1, 10001), 
                       nrows=1000)

# Get last 1000 rows
total_rows = 50000  # if you know total rows
sample_3 = pd.read_csv('big_data.csv', 
                       skiprows=range(1, total_rows-999))
```

**Visual Representation:**
```
Original File (10 rows):
Row 0: [Metadata]
Row 1: [Metadata]
Row 2: [HEADER]
Row 3: [Data 1]
Row 4: [Data 2]
Row 5: [Data 3]
Row 6: [Data 4]
Row 7: [Data 5]
Row 8: [Data 6]
Row 9: [Data 7]

skiprows=2, nrows=3:
Row 2: [HEADER]
Row 3: [Data 1]  ← Loaded
Row 4: [Data 2]  ← Loaded
Row 5: [Data 3]  ← Loaded
```

---

## 11. The `encoding` Parameter

**What is encoding?**  
Encoding is how computers store text. Different languages and systems use different encodings. If you see weird characters (like �), you probably have an encoding issue!

**Common encodings:**
- `utf-8` - Universal, works for most languages (DEFAULT)
- `latin-1` (or `iso-8859-1`) - Western European languages
- `cp1252` - Windows default for English
- `ascii` - Basic English only

```python
# Default encoding (usually works)
df = pd.read_csv('data.csv')

# Specify UTF-8 (best for international data)
df = pd.read_csv('data.csv', encoding='utf-8')

# For files created in Windows
df = pd.read_csv('data.csv', encoding='cp1252')

# For Latin characters
df = pd.read_csv('data.csv', encoding='latin-1')
```

**When you see errors:**
```python
# This might fail:
df = pd.read_csv('spanish_data.csv')
# UnicodeDecodeError: 'utf-8' codec can't decode byte...

# Try different encodings:
try:
    df = pd.read_csv('spanish_data.csv', encoding='utf-8')
except:
    try:
        df = pd.read_csv('spanish_data.csv', encoding='latin-1')
    except:
        df = pd.read_csv('spanish_data.csv', encoding='cp1252')
```

**Example with special characters:**
```python
# File contains: José, François, Müller
df = pd.read_csv('names.csv', encoding='utf-8')
print(df['name'].tolist())
# Output: ['José', 'François', 'Müller'] ✅

# Wrong encoding:
df = pd.read_csv('names.csv', encoding='ascii')
# Error or output: ['Jos�', 'Fran�ois', 'M�ller'] ❌
```

**How to detect encoding:**
```python
import chardet

# Read first few bytes to detect encoding
with open('unknown_encoding.csv', 'rb') as file:
    result = chardet.detect(file.read(10000))
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Use detected encoding
df = pd.read_csv('unknown_encoding.csv', encoding=encoding)
```

---

## 12. Handling Bad Lines

**What are bad lines?**  
Sometimes CSV files have rows with wrong number of columns, extra commas, or formatting errors.

### `on_bad_lines` Parameter (Pandas 1.3+)

```python
# Skip bad lines and continue
df = pd.read_csv('messy_data.csv', on_bad_lines='skip')

# Warn about bad lines but continue
df = pd.read_csv('messy_data.csv', on_bad_lines='warn')

# Stop on error (default)
df = pd.read_csv('messy_data.csv', on_bad_lines='error')
```

**Example Problematic CSV:**
```
name,age,city
John,25,NYC
Jane,30,LA,California,USA  <- Extra columns!
Mike,28,Chicago
Bob,35  <- Missing column!
```

```python
# This will fail:
df = pd.read_csv('bad_data.csv')
# ParserError: Expected 3 fields in line 3, saw 5

# Skip bad lines:
df = pd.read_csv('bad_data.csv', on_bad_lines='skip')
print(df)
# Output: Only loads John and Mike rows
```

### `error_bad_lines` (Older pandas versions)

```python
# For pandas < 1.3
df = pd.read_csv('messy_data.csv', 
                 error_bad_lines=False,  # Don't raise error
                 warn_bad_lines=True)    # Show warning
```

**Custom bad line handler:**
```python
def handle_bad_line(bad_line):
    print(f"Skipping bad line: {bad_line}")
    return None

df = pd.read_csv('messy_data.csv', 
                 on_bad_lines=handle_bad_line,
                 engine='python')  # Required for custom handler
```

**Best Practice:**
```python
# First, try to load and see what errors occur
try:
    df = pd.read_csv('suspicious_file.csv')
except Exception as e:
    print(f"Error: {e}")
    # Then load with skip
    df = pd.read_csv('suspicious_file.csv', on_bad_lines='skip')
    print(f"Loaded {len(df)} valid rows")
```

---

## 13. The `dtype` Parameter

**What is dtype?**  
`dtype` is "data type" - it tells pandas what kind of data each column contains (numbers, text, etc.).

**Common dtypes:**
- `int64` - Whole numbers
- `float64` - Decimal numbers
- `object` - Text/strings
- `bool` - True/False
- `datetime64` - Dates and times
- `category` - Categories (saves memory!)

### Checking dtypes

```python
df = pd.read_csv('aug_train.csv')

# Check all column types
print(df.dtypes)

# Check specific column
print(df['city'].dtype)

# Get detailed info
df.info()
```

**Example Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19158 entries, 0 to 19157
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   enrollee_id             19158 non-null  int64  
 1   city                    19158 non-null  object 
 2   city_development_index  19158 non-null  float64
 3   gender                  14650 non-null  object 
 4   education_level         18698 non-null  object 
```

### Specifying dtypes while loading

```python
# Specify dtype for specific columns
df = pd.read_csv('aug_train.csv', 
                 dtype={'enrollee_id': str,  # Load as string
                        'gender': 'category',  # Load as category
                        'training_hours': int})

# Specify dtype for all columns
df = pd.read_csv('data.csv', dtype=str)  # Everything as string

# Use dictionary for multiple columns
dtype_dict = {
    'id': str,
    'age': int,
    'salary': float,
    'department': 'category'
}
df = pd.read_csv('employees.csv', dtype=dtype_dict)
```

**Why specify dtypes?**

1. **Save Memory:**
```python
# Default: gender as object
df = pd.read_csv('aug_train.csv')
print(df['gender'].memory_usage())  # 153,264 bytes

# As category
df = pd.read_csv('aug_train.csv', dtype={'gender': 'category'})
print(df['gender'].memory_usage())  # 19,478 bytes
# 87% less memory! 😊
```

2. **Prevent Wrong Interpretation:**
```python
# ID column loaded as number
df = pd.read_csv('data.csv')
print(df['id'].dtype)  # int64
# Problem: 007 becomes 7, loses leading zero!

# Load ID as string
df = pd.read_csv('data.csv', dtype={'id': str})
print(df['id'].head())  # '007', '008', '009'  ✅
```

3. **Faster Operations:**
```python
# Category dtype is faster for grouping/filtering
df['department'] = df['department'].astype('category')
# 3-5x faster for groupby operations!
```

**Handling dtype errors:**
```python
# This might fail if column has mixed types
df = pd.read_csv('data.csv', dtype={'age': int})
# ValueError: invalid literal for int()

# Solution: Load as object, then convert
df = pd.read_csv('data.csv', dtype={'age': object})
df['age'] = pd.to_numeric(df['age'], errors='coerce')
# Converts invalid values to NaN
```

---

## 14. Handling Dates

**The Problem:**  
Pandas usually reads dates as text (object type), not as actual dates. This means you can't do date calculations!

```python
df = pd.read_csv('aug_train.csv')
print(df['last_new_job'].dtype)  # object (text)

# Can't do this:
# df['last_new_job'].max()  # Doesn't work properly!
```

### Converting Strings to Datetime

```python
import pandas as pd

# Sample data with date strings
data = {
    'employee_id': [1, 2, 3, 4],
    'join_date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05'],
    'last_login': ['2024-10-01 14:30:00', '2024-10-02 09:15:00', 
                   '2024-10-01 18:45:00', '2024-10-03 11:20:00']
}
df = pd.DataFrame(data)

print("Before conversion:")
print(df.dtypes)
# join_date      object  ← Text!
# last_login     object  ← Text!

# Convert to datetime
df['join_date'] = pd.to_datetime(df['join_date'])
df['last_login'] = pd.to_datetime(df['last_login'])

print("\nAfter conversion:")
print(df.dtypes)
# join_date      datetime64[ns]  ← Actual date!
# last_login     datetime64[ns]  ← Actual date!
```

### Different Date Formats

```python
# American format: MM/DD/YYYY
dates_us = ['01/15/2024', '02/20/2024']
df['date_us'] = pd.to_datetime(dates_us, format='%m/%d/%Y')

# European format: DD/MM/YYYY
dates_eu = ['15/01/2024', '20/02/2024']
df['date_eu'] = pd.to_datetime(dates_eu, format='%d/%m/%Y')

# Custom format: DD-Mon-YYYY
dates_custom = ['15-Jan-2024', '20-Feb-2024']
df['date_custom'] = pd.to_datetime(dates_custom, format='%d-%b-%Y')

# ISO format (automatic detection)
dates_iso = ['2024-01-15', '2024-02-20']
df['date_iso'] = pd.to_datetime(dates_iso)  # No format needed!
```

**Common format codes:**
```
%Y - 4-digit year (2024)
%y - 2-digit year (24)
%m - Month as number (01-12)
%d - Day of month (01-31)
%H - Hour 24-hour (00-23)
%I - Hour 12-hour (01-12)
%M - Minute (00-59)
%S - Second (00-59)
%b - Month abbreviation (Jan, Feb)
%B - Full month name (January)
```

### Handling Invalid Dates

```python
# Data with some invalid dates
dates_mixed = ['2024-01-15', '2024-13-45', 'invalid', '2024-03-10']

# errors='coerce' - Invalid dates become NaT (Not a Time)
df['date'] = pd.to_datetime(dates_mixed, errors='coerce')
print(df['date'])
# 0   2024-01-15
# 1          NaT  ← Invalid!
# 2          NaT  ← Invalid!
# 3   2024-03-10

# errors='ignore' - Keep original strings if parsing fails
df['date'] = pd.to_datetime(dates_mixed, errors='ignore')

# errors='raise' - Stop and raise error (default)
```

### Date Operations After Conversion

```python
df['join_date'] = pd.to_datetime(df['join_date'])

# Extract components
df['year'] = df['join_date'].dt.year
df['month'] = df['join_date'].dt.month
df['day'] = df['join_date'].dt.day
df['day_of_week'] = df['join_date'].dt.day_name()

# Calculate differences
df['days_since_join'] = (pd.Timestamp.now() - df['join_date']).dt.days

# Filter by date
recent = df[df['join_date'] > '2024-01-01']

print(df[['join_date', 'year', 'month', 'day_of_week', 'days_since_join']])
```

**Output:**
```
   join_date  year  month day_of_week  days_since_join
0 2024-01-15  2024      1      Monday              281
1 2024-02-20  2024      2    Tuesday              245
2 2024-03-10  2024      3      Sunday              227
```

---

## 15. The `parse_dates` Parameter

**What is `parse_dates`?**  
Instead of loading dates as strings and converting later, `parse_dates` converts them to datetime **while loading**. Much faster and convenient!

### Basic Usage

```python
# Convert single column to datetime
df = pd.read_csv('data.csv', parse_dates=['join_date'])

# Convert multiple columns
df = pd.read_csv('data.csv', 
                 parse_dates=['join_date', 'last_login', 'birth_date'])

# Parse all columns that look like dates (automatic)
df = pd.read_csv('data.csv', parse_dates=True)
```

**Example CSV:**
```
employee_id,name,join_date,last_login
1,John,2024-01-15,2024-10-01 14:30:00
2,Jane,2024-02-20,2024-10-02 09:15:00
```

```python
# WITHOUT parse_dates
df = pd.read_csv('employees.csv')
print(df.dtypes)
# join_date     object  ← String!
# last_login    object  ← String!

# WITH parse_dates
df = pd.read_csv('employees.csv', 
                 parse_dates=['join_date', 'last_login'])
print(df.dtypes)
# join_date     datetime64[ns]  ← Date!
# last_login    datetime64[ns]  ← Date!
```

### Combining Multiple Columns into One Date

```python
# CSV with separate date columns:
# year,month,day,temperature
# 2024,1,15,72
# 2024,1,16,68

# Combine year, month, day into single date
df = pd.read_csv('weather.csv', 
                 parse_dates={'date': ['year', 'month', 'day']})

print(df)
#         date  temperature
# 0 2024-01-15           72
# 1 2024-01-16           68
```

**Another example:**
```python
# CSV: date,time,value
# 2024-01-15,14:30:00,100

# Combine date and time columns
df = pd.read_csv('data.csv', 
                 parse_dates={'datetime': ['date', 'time']})
# Creates single 'datetime' column
```

### Custom Date Parser

```python
# For non-standard date formats
from datetime import datetime

def custom_date_parser(date_string):
    """Parse dates in format: 15-Jan-2024"""
    return datetime.strptime(date_string, '%d-%b-%Y')

df = pd.read_csv('data.csv', 
                 parse_dates=['custom_date'],
                 date_parser=custom_date_parser)
```

### Setting Date as Index

```python
# Parse date AND set it as index
df = pd.read_csv('timeseries.csv', 
                 parse_dates=['timestamp'],
                 index_col='timestamp')

print(df.index)
# DatetimeIndex(['2024-01-15', '2024-01-16', ...])
```

**Visual Comparison:**

```
WITHOUT parse_dates:
Read CSV → All columns as strings → Manually convert → Use dates
   (Slow, 2 steps)

WITH parse_dates:
Read CSV → Dates automatically converted → Use dates immediately
   (Fast, 1 step!)
```

---

## 16. The `converters` Parameter

**What is `converters`?**  
`converters` lets you apply custom functions to columns **while loading** the data. It's like preprocessing on-the-fly!

### Basic Usage

```python
# Define converter functions
def clean_name(name):
    """Convert name to title case and remove extra spaces"""
    return name.strip().title()

def extract_city_code(city):
    """Extract numeric code from 'city_103' → 103"""
    return int(city.split('_')[1])

def categorize_age(age):
    """Convert age to category"""
    age = int(age)
    if age < 25:
        return 'Young'
    elif age < 40:
        return 'Middle'
    else:
        return 'Senior'

# Apply converters while loading
df = pd.read_csv('data.csv', 
                 converters={
                     'name': clean_name,
                     'city': extract_city_code,
                     'age': categorize_age
                 })
```

**Example CSV:**
```
name,age,city
  john doe  ,23,city_103
JANE SMITH,35,city_040
  bob jones,45,city_162
```

**Without converters:**
```python
df = pd.read_csv('data.csv')
print(df)
#            name age      city
# 0   john doe   23  city_103
# 1  JANE SMITH  35  city_040
# 2   bob jones  45  city_162
```

**With converters:**
```python
df = pd.read_csv('data.csv', 
                 converters={
                     'name': clean_name,
                     'age': categorize_age,
                     'city': extract_city_code
                 })
print(df)
#          name      age  city
# 0   John Doe    Young   103
# 1 Jane Smith   Middle    40
# 2  Bob Jones   Senior   162
```

### Practical Examples

**1. Converting currency strings to numbers:**
```python
def parse_currency(value):
    """Convert '$1,234.56' to 1234.56"""
    return float(value.replace(', '').replace(',', ''))

df = pd.read_csv('sales.csv', 
                 converters={'price': parse_currency})

# Before: '$1,234.56' (string)
# After:  1234.56 (float)
```

**2. Normalizing phone numbers:**
```python
def normalize_phone(phone):
    """Convert various formats to standard: (555) 123-4567"""
    # Remove all non-numeric characters
    digits = ''.join(filter(str.isdigit, str(phone)))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

df = pd.read_csv('contacts.csv', 
                 converters={'phone': normalize_phone})

# Before: '555-123-4567', '5551234567', '(555)1234567'
# After:  '(555) 123-4567' (all standardized)
```

**3. Handling encoded values:**
```python
def decode_gender(code):
    """Convert M/F codes to full words"""
    mapping = {'M': 'Male', 'F': 'Female', 'm': 'Male', 'f': 'Female'}
    return mapping.get(str(code).strip(), 'Unknown')

df = pd.read_csv('people.csv', 
                 converters={'gender': decode_gender})

# Before: 'M', 'F', 'm', 'f'
# After:  'Male', 'Female', 'Male', 'Female'
```

**4. Creating features while loading:**
```python
def create_email(row_dict):
    """Create email from first and last name"""
    first = row_dict.get('first_name', '').lower()
    last = row_dict.get('last_name', '').lower()
    return f"{first}.{last}@company.com"

# Note: For row-wise operations, use lambda with multiple columns
df = pd.read_csv('employees.csv')
df['email'] = df.apply(lambda row: create_email(row), axis=1)
```

**5. Handling boolean values:**
```python
def parse_boolean(value):
    """Convert various formats to boolean"""
    true_values = ['yes', 'y', 'true', 't', '1', 1, True]
    return str(value).lower().strip() in true_values

df = pd.read_csv('survey.csv', 
                 converters={'subscribed': parse_boolean})

# Before: 'Yes', 'yes', 'Y', '1', 'True'
# After:  True, True, True, True, True
```

### Why Use Converters?

**Benefits:**
- ✅ Clean data while loading (one step)
- ✅ Saves memory (don't store intermediate results)
- ✅ Reusable functions for multiple datasets
- ✅ Automatic processing for large files

**When to use converters vs. post-processing:**

**Use converters when:**
- You need to apply same transformations every time you load the file
- You want to save memory
- Transformations are simple and column-specific

**Use post-processing when:**
- Transformations depend on multiple columns
- You need to see raw data first
- Complex operations requiring full DataFrame context

---

## 17. The `na_values` Parameter

**What is `na_values`?**  
`na_values` tells pandas which values should be treated as "missing" or "NaN" (Not a Number). By default, pandas recognizes some common ones, but you can add your own!

### Default NA Values

Pandas automatically treats these as NaN:
- Empty cells
- `NaN`, `nan`, `NA`, `N/A`, `null`
- `#N/A`, `-NaN`, `-nan`
- `None`

```python
df = pd.read_csv('data.csv')
# All default NA values are automatically converted to NaN
```

### Adding Custom NA Values

```python
# CSV content:
# name,age,gender,score
# John,25,Male,85
# Jane,-,Female,90
# Bob,30,Unknown,N/A
# Alice,28,Male,-999

# Specify custom NA values
df = pd.read_csv('data.csv', 
                 na_values=['-', 'Unknown', -999])

print(df)
#     name   age  gender  score
# 0   John  25.0    Male   85.0
# 1   Jane   NaN  Female   90.0
# 2    Bob  30.0     NaN    NaN
# 3  Alice  28.0    Male    NaN
```

### Column-Specific NA Values

```python
# Different NA values for different columns
custom_na = {
    'age': ['-', '?', 0],           # 0 is invalid age
    'gender': ['Unknown', 'N/A', 'Prefer not to say'],
    'score': [-999, -1],            # Sentinel values
    'email': ['none', 'n/a', '@']   # Invalid emails
}

df = pd.read_csv('data.csv', na_values=custom_na)
```

**Example:**
```python
# CSV:
# name,age,income,department
# John,25,50000,Sales
# Jane,?,--,Marketing
# Bob,0,0,--

# Define what's missing for each column
na_dict = {
    'age': ['?', 0],
    'income': ['--', 0],
    'department': ['--', 'Unknown']
}

df = pd.read_csv('employees.csv', na_values=na_dict)
print(df)
#    name   age  income department
# 0  John  25.0 50000.0      Sales
# 1  Jane   NaN     NaN  Marketing
# 2   Bob   NaN     NaN        NaN
```

### Combining with keep_default_na

```python
# Use only YOUR custom NA values (ignore defaults)
df = pd.read_csv('data.csv', 
                 na_values=['Missing', 'Unknown'],
                 keep_default_na=False)

# Now 'NA' will be kept as string 'NA', not converted to NaN!
```

### Real-World Examples

**1. Survey Data:**
```python
# CSV:
# question1,question2,question3
# Agree,Strongly Agree,Neutral
# Disagree,--,Prefer not to answer
# --,Agree,--

na_values = ['--', 'Prefer not to answer', 'N/A', 'Skip']

df = pd.read_csv('survey.csv', na_values=na_values)
```

**2. Sensor Data with Error Codes:**
```python
# CSV:
# timestamp,temperature,humidity
# 2024-01-15,72.5,45.2
# 2024-01-16,-999,ERROR
# 2024-01-17,73.1,-999

na_values = [-999, 'ERROR', 'FAULT', 'SENSOR_FAIL']

df = pd.read_csv('sensor_data.csv', na_values=na_values)
```

**3. Database Export with Special Codes:**
```python
# CSV:
# product_id,price,stock,location
# 101,29.99,50,A1
# 102,NULL,0,NULL
# 103,--,--,B2

na_values = {
    'price': ['NULL', '--', 0.0],
    'stock': ['--', 'NULL'],
    'location': ['NULL', '--', 'TBD']
}

df = pd.read_csv('inventory.csv', na_values=na_values)
```

**4. Mixed Missing Indicators:**
```python
# Different departments use different missing codes
na_values = [
    'N/A', 'NA', 'n/a', 'na',       # Common
    'missing', 'Missing', 'MISSING', # Text
    '?', '--', '---',                # Symbols
    'null', 'NULL', 'Null',          # Database
    '', ' ', '  ',                   # Empty/spaces
    -999, -9999, 9999,               # Sentinel numbers
    'Unknown', 'unknown', 'UNKNOWN'  # Unknown values
]

df = pd.read_csv('messy_data.csv', na_values=na_values)
```

### Checking for Missing Values

```python
df = pd.read_csv('data.csv', na_values=['-', '?', 'Unknown'])

# Count missing values per column
print(df.isnull().sum())

# Percentage of missing values
print((df.isnull().sum() / len(df)) * 100)

# Show rows with any missing values
print(df[df.isnull().any(axis=1)])

# Drop rows with missing values
df_clean = df.dropna()

# Fill missing values
df_filled = df.fillna({'age': 0, 'gender': 'Unknown'})
```

**Visual Representation:**
```
CSV File:
name    age   score
John     25      85
Jane      -      90
Bob      30   -999

Without na_values:
name    age   score
John     25      85
Jane      -      90  ← String "-"
Bob      30   -999  ← Actual -999

With na_values=['-', -999]:
name    age   score
John   25.0    85.0
Jane    NaN    90.0  ← Now NaN
Bob    30.0     NaN  ← Now NaN
```

---

## 18. Loading Large Datasets in Chunks

**The Problem:**  
Large CSV files (100MB+) can:
- Take forever to load
- Crash your computer (not enough RAM)
- Make your code slow

**The Solution: Chunking!**  
Load the file in small pieces (chunks), process each piece, then combine results.

### Basic Chunking

```python
# Load in chunks of 1000 rows
chunk_size = 1000
chunks = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # Process each chunk
    chunks.append(chunk)

# Combine all chunks
df = pd.concat(chunks, ignore_index=True)
```

**Visual Representation:**
```
Huge File (1,000,000 rows):
    ⬇️ chunksize=1000
[Chunk 1: Rows 0-999]     → Process → Store
[Chunk 2: Rows 1000-1999] → Process → Store
[Chunk 3: Rows 2000-2999] → Process → Store
...
[Chunk 1000: Rows 999000-999999] → Process → Store
    ⬇️
Combine all results
```

### Processing While Loading

```python
# Only keep rows that meet criteria
chunk_size = 5000
filtered_data = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # Filter each chunk
    filtered = chunk[chunk['age'] > 25]
    filtered_data.append(filtered)

# Combine filtered chunks
df = pd.concat(filtered_data, ignore_index=True)
print(f"Filtered from millions to {len(df)} rows")
```

### Aggregating Data

```python
# Calculate statistics without loading entire file
chunk_size = 10000
total_sum = 0
total_count = 0

for chunk in pd.read_csv('sales_data.csv', chunksize=chunk_size):
    total_sum += chunk['revenue'].sum()
    total_count += len(chunk)

average_revenue = total_sum / total_count
print(f"Average revenue: ${average_revenue:.2f}")
```

### Real-World Example: Processing Transaction Data

```python
# CSV: huge_transactions.csv (10GB file)
# transaction_id,date,customer_id,amount,product

chunk_size = 50000
high_value_transactions = []
monthly_totals = {}

print("Processing large transaction file...")

for i, chunk in enumerate(pd.read_csv('huge_transactions.csv', 
                                       chunksize=chunk_size)):
    # Progress indicator
    print(f"Processing chunk {i+1}...")
    
    # Find high-value transactions (>$10,000)
    high_value = chunk[chunk['amount'] > 10000]
    high_value_transactions.append(high_value)
    
    # Calculate monthly totals
    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk['month'] = chunk['date'].dt.to_period('M')
    monthly = chunk.groupby('month')['amount'].sum()
    
    for month, total in monthly.items():
        if month in monthly_totals:
            monthly_totals[month] += total
        else:
            monthly_totals[month] = total

# Combine results
high_value_df = pd.concat(high_value_transactions, ignore_index=True)
print(f"\nFound {len(high_value_df)} high-value transactions")
print(f"Processed {sum(len(chunk) for chunk in high_value_transactions)} total rows")
```

### Chunking with Filtering and Transformation

```python
def process_chunk(chunk):
    """Process each chunk: clean, transform, filter"""
    # Clean data
    chunk = chunk.dropna(subset=['email'])
    
    # Transform
    chunk['name'] = chunk['name'].str.title()
    chunk['age_group'] = pd.cut(chunk['age'], 
                                 bins=[0, 18, 30, 50, 100],
                                 labels=['Teen', 'Young', 'Middle', 'Senior'])
    
    # Filter
    chunk = chunk[chunk['country'] == 'USA']
    
    return chunk

# Process file in chunks
chunk_size = 10000
processed_chunks = []

for chunk in pd.read_csv('customer_data.csv', chunksize=chunk_size):
    processed = process_chunk(chunk)
    processed_chunks.append(processed)

final_df = pd.concat(processed_chunks, ignore_index=True)
```

### Saving Chunks to Multiple Files

```python
# Split large file into smaller files
chunk_size = 100000
file_number = 1

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    output_file = f'split_file_{file_number}.csv'
    chunk.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    file_number += 1
```

### Memory-Efficient Statistical Analysis

```python
# Calculate statistics without loading full file
chunk_size = 50000

# Initialize accumulators
count = 0
sum_values = 0
sum_squares = 0
min_value = float('inf')
max_value = float('-inf')

for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
    count += len(chunk)
    sum_values += chunk['value'].sum()
    sum_squares += (chunk['value'] ** 2).sum()
    min_value = min(min_value, chunk['value'].min())
    max_value = max(max_value, chunk['value'].max())

# Calculate final statistics
mean = sum_values / count
variance = (sum_squares / count) - (mean ** 2)
std_dev = variance ** 0.5

print(f"Rows processed: {count:,}")
print(f"Mean: {mean:.2f}")
print(f"Std Dev: {std_dev:.2f}")
print(f"Min: {min_value:.2f}")
print(f"Max: {max_value:.2f}")
```

### Progress Bar with Chunks

```python
from tqdm import tqdm
import os

# Get file size to estimate chunks
file_size = os.path.getsize('large_file.csv')
chunk_size = 10000

# Estimate number of chunks (approximate)
estimated_chunks = file_size // (chunk_size * 100)  # Rough estimate

results = []
with tqdm(total=estimated_chunks, desc="Processing") as pbar:
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        # Process chunk
        processed = chunk[chunk['score'] > 50]
        results.append(processed)
        pbar.update(1)

final_df = pd.concat(results, ignore_index=True)
```

### When to Use Chunking

**Use chunking when:**
- ✅ File size > 1GB
- ✅ File won't fit in RAM
- ✅ You only need to process subset of data
- ✅ You're doing aggregations (sum, mean, count)
- ✅ You want to filter data before loading fully

**Don't use chunking when:**
- ❌ File is small (<100MB)
- ❌ You need to operate on entire dataset at once
- ❌ Operations require seeing all rows together
- ❌ You have enough RAM

### Optimal Chunk Size

```python
# Too small (slower)
chunksize = 100  # Too many iterations

# Too large (might run out of memory)
chunksize = 10000000  # Defeats the purpose

# Just right (balance between memory and speed)
chunksize = 10000  # For most cases
chunksize = 50000  # For systems with more RAM
chunksize = 100000  # For very large RAM systems

# Rule of thumb: Each chunk should be 50-100 MB
```

---

## 🎯 Complete Example: Real-World ML Dataset Preparation

Let's put everything together with a comprehensive example!

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
file_path = 'aug_train.csv'
chunk_size = 5000

# Custom functions
def clean_city_code(city):
    """Extract numeric code from city_XXX format"""
    if pd.isna(city):
        return np.nan
    return int(city.split('_')[1])

def categorize_experience(years):
    """Convert experience to categories"""
    if pd.isna(years):
        return 'Unknown'
    years = int(years)
    if years < 2:
        return 'Junior'
    elif years < 5:
        return 'Mid'
    elif years < 10:
        return 'Senior'
    else:
        return 'Expert'

# Load and process in chunks
processed_chunks = []

print("Loading and processing data...")

for i, chunk in enumerate(pd.read_csv(
    file_path,
    chunksize=chunk_size,
    dtype={
        'enrollee_id': str,
        'gender': 'category',
        'enrolled_university': 'category',
        'education_level': 'category',
        'major_discipline': 'category',
        'experience': str,
        'company_size': 'category',
        'company_type': 'category',
        'last_new_job': str
    },
    na_values=['', ' ', 'NA', 'N/A', 'null', 'NULL', '-', '?'],
    parse_dates=False,  # We'll handle manually
    usecols=lambda col: col != 'irrelevant_column',  # Skip unwanted columns
    converters={'city': clean_city_code}
)):
    
    print(f"  Processing chunk {i+1}...")
    
    # Handle experience
    chunk['experience_category'] = chunk['experience'].apply(
        lambda x: categorize_experience(x) if pd.notna(x) else 'Unknown'
    )
    
    # Create features
    chunk['has_relevant_exp'] = chunk['relevent_experience'].notna()
    chunk['is_enrolled'] = chunk['enrolled_university'].notna()
    
    # Filter: Keep only relevant rows
    chunk = chunk[chunk['training_hours'] > 0]
    
    processed_chunks.append(chunk)

# Combine all chunks
df = pd.concat(processed_chunks, ignore_index=True)

print(f"\n✅ Successfully loaded and processed {len(df)} rows")
print(f"\nDataset Info:")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nMissing values:")
print(df.isnull().sum())

# Save processed data
df.to_csv('processed_training_data.csv', index=False)
print("\n💾 Saved processed data to 'processed_training_data.csv'")
```

---

## 📚 Quick Reference Cheat Sheet

```python
# Basic loading
df = pd.read_csv('file.csv')

# With all common parameters
df = pd.read_csv(
    'file.csv',
    sep=',',                              # Separator
    header=0,                             # Header row
    names=['col1', 'col2'],               # Custom column names
    index_col='id',                       # Use column as index
    usecols=['col1', 'col2'],            # Load only these columns
    dtype={'col1': str, 'col2': int},    # Specify data types
    parse_dates=['date_col'],             # Convert to datetime
    na_values=['-', '?', 'Unknown'],     # Custom missing values
    skiprows=2,                           # Skip first 2 rows
    nrows=1000,                           # Load only 1000 rows
    encoding='utf-8',                     # File encoding
    chunksize=5000,                       # Load in chunks
    on_bad_lines='skip',                  # Skip problematic rows
    converters={'col': custom_function}   # Apply functions while loading
)
```

---

## 🚀 Performance Tips

1. **Use appropriate dtypes** - Save up to 80% memory
2. **Load only needed columns** - Use `usecols`
3. **Use chunking for large files** - Prevents memory issues
4. **Use category dtype** - For columns with few unique values
5. **Specify parse_dates** - Faster than converting after loading
6. **Use converters** - Transform while loading
7. **Skip bad lines** - Don't let errors stop the whole load

---

## 🎓 Summary

You've learned:
- ✅ How to import and use pandas
- ✅ Loading CSV files from local and URLs
- ✅ All major `pd.read_csv()` parameters
- ✅ Handling dates, missing values, and data types
- ✅ Processing large files efficiently with chunking
- ✅ Real-world data preprocessing techniques

**Next Steps:**
1. Practice with your own datasets
2. Experiment with different parameter combinations
3. Learn data cleaning and visualization
4. Explore machine learning with your loaded data!

---

## 📖 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Real Python Pandas Tutorials](https://realpython.com/pandas-python-explore-dataset/)

---

<div align="center">

**Happy Data Loading! 🐼📊**

*Made with ❤️ for aspiring Machine Learning Engineers*

</div>
