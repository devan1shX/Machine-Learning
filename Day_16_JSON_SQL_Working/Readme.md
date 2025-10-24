# 🚀 Complete Guide: Working with JSON & SQL Data in Python for Machine Learning

> A beginner-friendly guide to loading, exploring, and manipulating JSON and SQL datasets using Python and Pandas

---

## 📚 Table of Contents

1. [Introduction to JSON Data](#-introduction-to-json-data)
2. [Loading JSON Files with Pandas](#-loading-json-files-with-pandas)
3. [Loading JSON from URLs](#-loading-json-from-urls)
4. [Working with SQL Databases](#-working-with-sql-databases)
5. [Connecting to MySQL](#-connecting-to-mysql)
6. [Reading SQL Data with Pandas](#-reading-sql-data-with-pandas)
7. [Advanced Options](#-advanced-options)

---

## 🎯 Introduction to JSON Data

**JSON (JavaScript Object Notation)** is a lightweight format for storing and transporting data. Think of it like a dictionary in Python - it stores data in **key-value pairs**.

### Why JSON?
- **Human-readable**: Easy to read and write
- **Widely used**: APIs, web services, configuration files
- **Flexible**: Can store nested data structures

### JSON Structure Example:
```json
{
  "id": 1,
  "name": "Pizza Margherita",
  "cuisine": "italian",
  "ingredients": ["tomato", "mozzarella", "basil"]
}
```

---

## 📂 Loading JSON Files with Pandas

Let's start by importing pandas - your best friend for data manipulation!

### Step 1: Import Pandas

```python
import pandas as pd
```

**What is Pandas?** It's a powerful library that makes working with data super easy. Think of it as Excel, but with superpowers! 💪

---

### Step 2: Reading a JSON File

Let's use the **Cuisine Dataset** as our example. This dataset contains recipes from different cuisines around the world.

```python
# Load the JSON file
df = pd.read_json('train.json')

# Display first 5 rows
print(df.head())
```

**Expected Output:**
```
   id          cuisine                                        ingredients
0   1          italian    [tomato, mozzarella, basil, olive oil]
1   2          mexican    [corn, black beans, avocado, lime]
2   3          chinese    [rice, soy sauce, ginger, garlic]
3   4           indian    [curry powder, coconut milk, chicken, rice]
4   5          italian    [pasta, tomato sauce, parmesan, garlic]
```

---

### Step 3: Understanding Key Parameters

Just like reading other file types, `pd.read_json()` has several useful parameters:

#### 🔹 **dtype** - Specify Data Types

```python
# Specify that 'id' should be integer and 'cuisine' should be string
df = pd.read_json('train.json', dtype={'id': int, 'cuisine': str})
```

**Why use dtype?**
- Saves memory (integers use less space than floats)
- Prevents errors (ensures columns have correct data type)
- Faster processing

---

#### 🔹 **convert_dates** - Parse Date Columns

```python
# If your JSON has date columns, convert them automatically
df = pd.read_json('train.json', convert_dates=['created_date', 'updated_date'])
```

**What does this do?**
- Converts date strings (like "2024-01-15") into proper datetime objects
- Makes it easy to filter by date, extract year/month, etc.

**Example Transformation:**
```
Before: "2024-01-15" (string)
After:  2024-01-15 00:00:00 (datetime object)
```

---

#### 🔹 **encoding** - Handle Special Characters

```python
# For datasets with special characters (like é, ñ, 中文)
df = pd.read_json('train.json', encoding='utf-8')
```

**Common encodings:**
- `utf-8` - Universal (works for most languages)
- `latin-1` - For Western European languages
- `cp1252` - Windows encoding

**Why important?** Without proper encoding, "café" might appear as "cafÃ©" 😱

---

#### 🔹 **chunksize** - Load Large Files in Pieces

```python
# Load 1000 rows at a time for huge files
chunk_iterator = pd.read_json('train.json', lines=True, chunksize=1000)

for chunk in chunk_iterator:
    print(f"Processing chunk with {len(chunk)} rows")
    # Process each chunk here
```

**Why use chunks?**
- Prevents memory errors with massive datasets
- Allows processing datasets larger than your RAM
- Faster initial load time

**Visual Example:**
```
Instead of loading:
[═══════════════════════] 1,000,000 rows at once ❌

Load in chunks:
[════] 1,000 rows ✓
[════] 1,000 rows ✓
[════] 1,000 rows ✓
... (continues)
```

---

#### 🔹 **nrows** - Load Limited Rows

```python
# Load only first 100 rows for quick testing
df = pd.read_json('train.json', lines=True, nrows=100)
```

**When to use?**
- Testing your code before processing the full dataset
- Quick data exploration
- Creating sample datasets

---

### Step 4: Exploring Your JSON Data

```python
# Check the shape (rows, columns)
print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")

# See column names and types
print(df.info())

# Basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
```

**Example Output:**
```
Dataset has 5 rows and 3 columns

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   id           5 non-null      int64 
 1   cuisine      5 non-null      object
 2   ingredients  5 non-null      object
```

---

## 🌐 Loading JSON from URLs

Often, you'll need to load data directly from the internet (like APIs or online datasets). Here's how!

### Example: Loading Currency Exchange Data

```python
import pandas as pd

# URL to JSON data
url = 'https://api.exchangerate-api.com/v4/latest/USD'

# Load JSON directly from URL
df = pd.read_json(url)

print(df)
```

**Expected Output:**
```
                    rates
EUR              0.85
GBP              0.73
INR             74.50
JPY            110.25
AUD              1.35
```

---

### Another Example: GitHub API

```python
# Load public GitHub repo data
url = 'https://api.github.com/repos/pandas-dev/pandas'
df = pd.read_json(url, typ='series')  # typ='series' for single record

print(f"Repository Name: {df['name']}")
print(f"Stars: {df['stargazers_count']}")
print(f"Language: {df['language']}")
```

---

### Template for Your Own Datasets

```python
# TEMPLATE: Use this for any JSON URL
import pandas as pd

# Replace with your URL
url = 'YOUR_JSON_URL_HERE'

# Load the data
df = pd.read_json(url)

# For JSON with nested structure, use:
# df = pd.read_json(url, orient='records')

# Display
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
```

---

## 💾 Working with SQL Databases

SQL (Structured Query Language) databases store data in **tables** - think of them like Excel spreadsheets, but way more powerful!

### Why SQL?
- **Structured data**: Data is organized in rows and columns
- **Relationships**: Tables can be linked together
- **Scalable**: Handle millions/billions of rows efficiently
- **Industry standard**: Used by most companies

---

## 🗄️ Example Dataset: World Cities Database

We'll use a popular Kaggle dataset containing information about cities worldwide.

**Dataset Structure:**
```
world (database)
├── city (table)
│   ├── ID
│   ├── Name
│   ├── CountryCode
│   └── Population
├── country (table)
│   ├── Code
│   ├── Name
│   └── Continent
```

---

## 🔌 Connecting to MySQL

### Step 1: Install MySQL Connector

First, install the connector library that allows Python to talk to MySQL databases:

```python
# Run this in Jupyter Notebook or terminal
!pip install mysql-connector-python
```

**What does this do?**
- Downloads and installs the MySQL connector library
- The `!` means "run this as a system command"
- Only need to do this ONCE per environment

---

### Step 2: Import the Library

```python
import mysql.connector
import pandas as pd
```

---

### Step 3: Download SQL Data File

If you have a `.sql` file from Kaggle:

1. **Download** the file (e.g., `world.sql`)
2. **Import** it into MySQL:

```bash
# In terminal or MySQL Workbench
mysql -u root -p < world.sql
```

Or use a GUI tool like **MySQL Workbench** or **phpMyAdmin** to import.

---

### Step 4: Establish Database Connection

```python
# Create connection
conn = mysql.connector.connect(
    host='localhost',        # Computer where MySQL runs (localhost = your computer)
    user='root',             # Your MySQL username
    password='your_password', # Your MySQL password
    database='world'         # Name of the database to connect to
)

print("✅ Connected to MySQL database!")
```

**Parameter Breakdown:**

| Parameter | What it means | Example |
|-----------|---------------|---------|
| `host` | Where is MySQL running? | `localhost` (your PC) or `192.168.1.100` (another computer) |
| `user` | Your MySQL username | `root`, `admin`, `john` |
| `password` | Your MySQL password | `MySecurePass123!` |
| `database` | Which database to use | `world`, `sales_db`, `customer_data` |

---

### Alternative: Using IP Address

If MySQL is on another computer:

```python
conn = mysql.connector.connect(
    host='192.168.1.100',  # IP address of the MySQL server
    port=3306,             # MySQL port (default is 3306)
    user='remote_user',
    password='password123',
    database='world'
)
```

---

## 📊 Reading SQL Data with Pandas

Now that we're connected, let's read data!

### Step 1: Basic Query

```python
# Write your SQL query
query = "SELECT * FROM city"

# Load data into pandas DataFrame
df = pd.read_sql_query(query, conn)

# Display first 5 rows
print(df.head())
```

**Expected Output:**
```
   ID           Name CountryCode  Population
0   1      Kabul         AFG     1780000
1   2  Qandahar         AFG      237500
2   3     Herat         AFG      186800
3   4  New York         USA     8175133
4   5    Mumbai         IND    12442373
```

---

### Step 2: Filtered Query

```python
# Get only cities with population > 1 million
query = """
SELECT Name, CountryCode, Population 
FROM city 
WHERE Population > 1000000
ORDER BY Population DESC
"""

df = pd.read_sql_query(query, conn)
print(df.head())
```

**Output:**
```
         Name CountryCode  Population
0  Mumbai         IND    12442373
1  Shanghai       CHN    11900000
2  Tokyo          JPN    11800000
3  New York       USA     8175133
4  Beijing        CHN     7480000
```

**Why this works:**
- `SELECT` → Choose which columns
- `FROM` → Which table
- `WHERE` → Filter rows
- `ORDER BY` → Sort results
- `DESC` → Descending order (biggest first)

---

### Step 3: Using Parameters

#### 🔹 **index_col** - Set Index Column

```python
# Use 'ID' column as the index
df = pd.read_sql_query(
    "SELECT * FROM city",
    conn,
    index_col='ID'
)

print(df.head())
```

**Before:**
```
   ID           Name CountryCode  Population
0   1      Kabul         AFG     1780000
```

**After:**
```
              Name CountryCode  Population
ID                                        
1         Kabul         AFG     1780000
```

**Why useful?** Makes it easy to access rows by ID: `df.loc[1]`

---

#### 🔹 **parse_dates** - Convert Date Columns

```python
# If you have date columns in your SQL table
query = "SELECT * FROM orders"

df = pd.read_sql_query(
    query,
    conn,
    parse_dates=['order_date', 'ship_date']
)
```

**What happens:**
```
Before: order_date = "2024-01-15" (string)
After:  order_date = 2024-01-15 00:00:00 (datetime)
```

**Now you can do:**
```python
# Filter by date
recent = df[df['order_date'] > '2024-01-01']

# Extract month
df['month'] = df['order_date'].dt.month
```

---

#### 🔹 **chunksize** - Load Large Tables in Chunks

```python
# Load 5000 rows at a time
chunk_iterator = pd.read_sql_query(
    "SELECT * FROM city",
    conn,
    chunksize=5000
)

# Process each chunk
for i, chunk in enumerate(chunk_iterator):
    print(f"Processing chunk {i+1}: {len(chunk)} rows")
    
    # Example: Calculate average population per chunk
    avg_pop = chunk['Population'].mean()
    print(f"Average population: {avg_pop:,.0f}\n")
```

**Visual Process:**
```
Table with 50,000 rows
↓
Chunk 1: Rows 1-5,000    ✓ Process
Chunk 2: Rows 5,001-10,000  ✓ Process
Chunk 3: Rows 10,001-15,000 ✓ Process
... (continues)
```

---

## 🎓 Advanced Options

### Multiple Queries

```python
# Query 1: Get cities
cities_query = "SELECT * FROM city WHERE Population > 1000000"
cities_df = pd.read_sql_query(cities_query, conn)

# Query 2: Get countries
countries_query = "SELECT * FROM country"
countries_df = pd.read_sql_query(countries_query, conn)

# Merge them together
merged = cities_df.merge(
    countries_df,
    left_on='CountryCode',
    right_on='Code'
)

print(merged[['Name_x', 'Population', 'Name_y', 'Continent']].head())
```

**Output:**
```
      Name_x  Population      Name_y     Continent
0     Mumbai    12442373       India          Asia
1   Shanghai    11900000       China          Asia
2      Tokyo    11800000       Japan          Asia
3   New York     8175133  United States  North America
4    Beijing     7480000       China          Asia
```

---

### Close Connection (Important!)

Always close your connection when done:

```python
# Close the connection
conn.close()
print("🔒 Connection closed!")
```

**Why?** Keeps your database secure and frees up resources.

---

## 🎯 Complete Example: End-to-End Workflow

```python
import pandas as pd
import mysql.connector

# 1. CONNECT TO DATABASE
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='your_password',
    database='world'
)

# 2. LOAD DATA
query = """
SELECT 
    city.Name AS City,
    city.Population,
    country.Name AS Country,
    country.Continent
FROM city
JOIN country ON city.CountryCode = country.Code
WHERE city.Population > 5000000
ORDER BY city.Population DESC
"""

df = pd.read_sql_query(
    query,
    conn,
    index_col='City',
    parse_dates=None
)

# 3. EXPLORE DATA
print(f"Loaded {len(df)} cities")
print(f"\nTop 5 most populous cities:")
print(df.head())

print(f"\nAverage population: {df['Population'].mean():,.0f}")
print(f"Total population: {df['Population'].sum():,.0f}")

# 4. ANALYZE BY CONTINENT
continent_stats = df.groupby('Continent')['Population'].agg(['count', 'sum', 'mean'])
print(f"\nCities by Continent:")
print(continent_stats)

# 5. CLOSE CONNECTION
conn.close()
print("\n✅ Done!")
```

---

## 🏆 Key Takeaways

### JSON
✅ Use `pd.read_json()` for local files  
✅ Load from URLs directly  
✅ Use `chunksize` for large files  
✅ Specify `dtype` to optimize memory  

### SQL
✅ Install `mysql-connector-python` first  
✅ Create connection with host, user, password, database  
✅ Use `pd.read_sql_query()` to load data  
✅ Always close connection when done  

### Common Parameters
- **index_col**: Set which column becomes the index
- **parse_dates**: Convert date strings to datetime objects
- **chunksize**: Load large data in manageable pieces
- **nrows**: Load only specified number of rows

---

## 📚 Next Steps

1. Practice with different datasets
2. Try complex SQL queries (JOIN, GROUP BY, HAVING)
3. Combine JSON and SQL data
4. Build your first machine learning model!

---

**Happy Learning! 🚀**

*Created with ❤️ for aspiring data scientists*
