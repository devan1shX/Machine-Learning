🐼 Learning Pandas in Python for EDA
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; margin: 20px 0;"> <h2 style="color: #ffffff; margin: 0; font-weight: 600;">What is Pandas?</h2> </div>

Pandas is a powerful, open-source data manipulation and analysis library for Python. Built on top of NumPy, it provides high-performance, easy-to-use data structures and tools specifically designed for working with structured data.

🎯 Core Functionality
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; margin: 15px 0;">

Pandas excels at:

Data Cleaning → Handle missing values, duplicates, and inconsistencies

Data Transformation → Reshape, pivot, merge, and aggregate datasets

Data Analysis → Statistical operations and time series analysis

Data Visualization → Quick plotting capabilities integrated with Matplotlib

</div>

✨ Key Benefits
<table style="width: 100%; border-collapse: collapse; margin: 20px 0;"> <tr style="background: linear-gradient(135deg, #3494e6 0%, #ec6ead 100%);"> <th style="padding: 15px; color: #ffffff; text-align: left; border-radius: 8px 0 0 0;">Benefit</th> <th style="padding: 15px; color: #ffffff; text-align: left; border-radius: 0 8px 0 0;">Description</th> </tr> <tr style="background: #1a1a2e;"> <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid #333;"><strong>⚡ Performance</strong></td> <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid #333;">Optimized C-based operations for handling large datasets efficiently</td> </tr> <tr style="background: #16213e;"> <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid #333;"><strong>🔄 Flexibility</strong></td> <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid #333;">Works seamlessly with CSV, Excel, SQL, JSON, and more</td> </tr> <tr style="background: #1a1a2e;"> <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid #333;"><strong>🎨 Intuitive</strong></td> <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid #333;">Pythonic syntax with DataFrame and Series structures</td> </tr> <tr style="background: #16213e;"> <td style="padding: 12px; color: #e0e0e0;"><strong>🌐 Integration</strong></td> <td style="padding: 12px; color: #e0e0e0;">Compatible with NumPy, Matplotlib, Scikit-learn ecosystem</td> </tr> </table>

📦 Installation
<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #ffd700; margin-top: 0;">💻 Installing on Jupyter Notebook</h3> <p style="color: #b0b0b0; margin-bottom: 15px;">Execute the following command in a code cell:</p> </div> </div>

Python

# Install pandas in Jupyter environment
!pip install pandas --upgrade
<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #4ecdc4; margin-top: 0;">🪟 Installing on Windows (Command Prompt/PowerShell)</h3> <p style="color: #b0b0b0; margin-bottom: 15px;">Open your terminal and run:</p> </div> </div>

Bash

# Standard installation
pip install pandas

# Or with specific version
pip install pandas==2.1.0

# Verify installation
python -c "import pandas as pd; print(pd.__version__)"
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 25px 0; border-left: 5px solid #ffd700;"> <p style="color: #ffffff; margin: 0; font-size: 14px;"> 💡 <strong>Pro Tip:</strong> After installation, restart your Jupyter kernel to ensure pandas is properly loaded into your environment. </p> </div>

<div style="text-align: center; margin: 30px 0; padding: 20px; background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); border-radius: 10px;"> <h3 style="color: #ffffff; margin: 0;">🚀 Ready to Start Your EDA Journey!</h3> </div>

Python

import pandas as pd
import numpy as np
📊 Creating Your First DataFrame
<div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #38ef7d; margin-top: 0;">🎯 DataFrame from Dictionary</h3> <p style="color: #b0b0b0; margin-bottom: 10px;">A DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. Let's create one from a Python dictionary:</p> </div> </div>

Python

import pandas as pd

# Create a dictionary with student data
student_data = {
    "name": ['Anish', 'Manish'],
    "marks": [92, 82],
    "city": ['Delhi', 'Mumbai']
}

# Convert dictionary to DataFrame
df = pd.DataFrame(student_data)

# Display the DataFrame
df
<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #38ef7d;"> <p style="color: #1e1e1e; margin: 0; font-size: 14px;"> 📝 <strong>Note:</strong> Each dictionary key becomes a column name, and the values (lists) become the rows. This is one of the most common ways to create DataFrames in pandas! </p> </div>

Python

dict1 = {
    "name" : ['Anish', 'Manish'],
    "makrs" : [92, 82],
    "city" : ['Delhi', 'Mumbai']
}

df = pd.DataFrame(dict1)
df
Plaintext

     name  makrs    city
0   Anish     92   Delhi
1  Manish     82  Mumbai
💾 Exporting DataFrame to CSV
<div style="background: linear-gradient(135deg, #7f00ff 0%, #e100ff 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #e100ff; margin-top: 0;">📁 Saving Data to CSV Files</h3> <p style="color: #b0b0b0; margin-bottom: 10px;">Learn how to export your DataFrame to CSV format with and without index column:</p> </div> </div>

Python

# Export DataFrame with default index column
df.to_csv("data.csv")

# Export DataFrame without index column
df.to_csv("data_without_index.csv", index=False)
<div style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 15px; border-radius: 8px;"> <h4 style="color: #feca57; margin-top: 0;">📄 Output: data.csv (with index)</h4> </div> </div>

Code snippet

,name,marks,city
0,Anish,92,Delhi
1,Manish,82,Mumbai
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 15px; border-radius: 8px;"> <h4 style="color: #00f2fe; margin-top: 0;">📄 Output: data_without_index.csv (without index)</h4> </div> </div>

Code snippet

name,marks,city
Anish,92,Delhi
Manish,82,Mumbai
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #00f2fe;"> <p style="color: #ffffff; margin: 0; font-size: 14px;"> 💡 <strong>Key Difference:</strong> Using <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">index=False</code> removes the default numeric index column from the CSV file, creating a cleaner output for sharing or importing into other tools. </p> </div>

Python

df.to_csv("data.csv")

df.to_csv("data_without_index", index = False)
🔍 Exploring DataFrame Methods
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #f5576c; margin-top: 0;">👀 Quick Data Preview & Statistical Summary</h3> <p style="color: #b0b0b0; margin-bottom: 10px;">Essential methods to inspect and understand your dataset at a glance:</p> </div> </div>

📌 View Top Rows with head()
Python

# Display first 5 rows (default)
df.head()
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #f093fb;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;"></th> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;">name</th> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;">marks</th> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

Python

# Display only first row
df.head(1)
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #f093fb;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;"></th> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;">name</th> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;">marks</th> <th style="padding: 10px; color: #f093fb; text-align: left; border-bottom: 2px solid #f5576c;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> </table> </div>

📌 View Bottom Rows with tail()
Python

# Display last 5 rows (default)
df.tail()
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #38ef7d;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;"></th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">name</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">marks</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

Python

# Display only last row
df.tail(1)
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #38ef7d;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;"></th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">name</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">marks</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

📊 Statistical Summary with describe()
Python

# Generate descriptive statistics for numerical columns
df.describe()
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #e100ff;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #e100ff; text-align: left; border-bottom: 2px solid #7f00ff;"></th> <th style="padding: 10px; color: #e100ff; text-align: center; border-bottom: 2px solid #7f00ff;">marks</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">count</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">2.0</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">mean</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">87.0</td> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">std</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">7.071068</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">min</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">82.0</td> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">25%</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">84.5</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">50%</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">87.0</td> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">75%</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">89.5</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">max</td> <td style="padding: 10px; color: #e0e0e0; text-align: center;">92.0</td> </tr> </table> </div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #e100ff;"> <p style="color: #ffffff; margin: 0; font-size: 14px;"> 💡 <strong>Pro Tip:</strong> <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">describe()</code> automatically calculates count, mean, standard deviation, min/max, and quartiles for all numerical columns—perfect for quick statistical analysis! </p> </div>

📥 Reading & Accessing DataFrame Data
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #00f2fe; margin-top: 0;">📂 Loading CSV Files & Accessing Elements</h3> <p style="color: #b0b0b0; margin-bottom: 10px;">Learn how to read CSV files and access specific columns and values from your DataFrame:</p> </div> </div>

📖 Reading CSV File
Python

# Load data from CSV file
data = pd.read_csv('data.csv')
data
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #4facfe;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">Unnamed: 0</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">name</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">marks</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

🎯 Accessing a Single Column
Python

# Access the 'name' column (returns a Series)
data['name']
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #f093fb;"> <pre style="color: #e0e0e0; margin: 0; font-family: 'Courier New', monospace;"> <span style="color: #feca57;">0</span> Anish <span style="color: #feca57;">1</span> Manish <span style="color: #38ef7d;">Name:</span> name, <span style="color: #38ef7d;">dtype:</span> object </pre> </div>

🔢 Accessing Specific Value by Index
Python

# Access the value at index 1 in 'name' column
data['name'][1]
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #38ef7d;"> <pre style="color: #e0e0e0; margin: 0; font-family: 'Courier New', monospace; font-size: 16px;"> <span style="color: #feca57;">'Manish'</span> </pre> </div>

<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #38ef7d;"> <p style="color: #1e1e1e; margin: 0; font-size: 14px;"> ⚠️ <strong>Note:</strong> The CSV was saved with index, creating an extra <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">'Unnamed: 0'</code> column. Use <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">index_col=0</code> parameter to avoid this: <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">pd.read_csv('data.csv', index_col=0)</code> </p> </div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4facfe;"> <p style="color: #ffffff; margin: 0; font-size: 14px;"> 💡 <strong>Pro Tip:</strong> Use <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">.loc[]</code> or <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">.iloc[]</code> for more robust indexing: <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">data.loc[1, 'name']</code> or <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">data.iloc[1, 0]</code> </p> </div>

✏️ Modifying DataFrame Values
<div style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #feca57; margin-top: 0;">🔧 Updating Values in DataFrame</h3> <p style="color: #b0b0b0; margin-bottom: 10px;">Learn the correct way to modify values in your DataFrame to avoid warnings:</p> </div> </div>

📋 Original DataFrame
Python

df
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #4facfe;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;"></th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">name</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">marks</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

❌ Incorrect Method (Chained Assignment)
Python

# This works but generates a warning (not recommended)
df['name'][1] = 'Rohan'
df
<div style="background: #2d1b1b; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #ff6b6b;"> <p style="color: #ff6b6b; margin: 0; font-weight: bold; font-size: 14px;">⚠️ FutureWarning & SettingWithCopyWarning</p> <pre style="color: #ffb3b3; margin: 10px 0 0 0; font-size: 12px; overflow-x: auto;"> ChainedAssignmentError: behaviour will change in pandas 3.0! You are setting values through chained assignment.

Use df.loc[row_indexer, "col"] = values instead. </pre>

</div>

<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #38ef7d;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;"></th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">name</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">marks</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #38ef7d;">Rohan</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

✅ Correct Method (Using .loc[])
Python

# Recommended approach - use .loc[] for safe assignment
df.loc[1, 'name'] = 'Rohan'
df
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #38ef7d;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;"></th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">name</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">marks</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #38ef7d;">Rohan</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #38ef7d;"> <h4 style="color: #ffffff; margin-top: 0;">📚 Why Use .loc[]?</h4> <ul style="color: #e0e0e0; margin: 10px 0 0 20px; line-height: 1.8;"> <li><strong>Single Operation:</strong> Updates happen in one step, not chained</li> <li><strong>No Warnings:</strong> Avoids FutureWarning and SettingWithCopyWarning</li> <li><strong>Guaranteed Update:</strong> Works correctly with Copy-on-Write in pandas 3.0+</li> <li><strong>Clear Intent:</strong> Explicitly specifies row and column for modification</li> </ul> </div>

<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #e100ff;"> <p style="color: #1e1e1e; margin: 0; font-size: 14px;"> 💡 <strong>Pro Tip:</strong> Always use <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">.loc[row, column]</code> for label-based indexing or <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">.iloc[row, column]</code> for position-based indexing when modifying DataFrame values! </p> </div>

🏷️ Customizing DataFrame Index
<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 3px; border-radius: 10px; margin: 20px 0;"> <div style="background: #1e1e1e; padding: 20px; border-radius: 8px;"> <h3 style="color: #4ecdc4; margin-top: 0;">🔖 Setting Custom Row Labels</h3> <p style="color: #b0b0b0; margin-bottom: 10px;">Replace default numeric indices with meaningful custom labels:</p> </div> </div>

📋 Original DataFrame (Default Index)
Python

data
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #4facfe;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">Unnamed: 0</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">name</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">marks</th> <th style="padding: 10px; color: #4facfe; text-align: left; border-bottom: 2px solid #00f2fe;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">0</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #feca57; font-weight: bold;">1</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

🎯 Setting Custom Index Labels
Python

# Assign custom labels to DataFrame rows
data.index = ['first', 'second']
data
<div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 2px solid #38ef7d;"> <table style="width: 100%; border-collapse: collapse;"> <tr style="background: #2d2d2d;"> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">Unnamed: 0</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">name</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">marks</th> <th style="padding: 10px; color: #38ef7d; text-align: left; border-bottom: 2px solid #11998e;">city</th> </tr> <tr style="background: #1a1a1a;"> <td style="padding: 10px; color: #f093fb; font-weight: bold;">first</td> <td style="padding: 10px; color: #e0e0e0;">Anish</td> <td style="padding: 10px; color: #e0e0e0;">92</td> <td style="padding: 10px; color: #e0e0e0;">Delhi</td> </tr> <tr style="background: #252525;"> <td style="padding: 10px; color: #f093fb; font-weight: bold;">second</td> <td style="padding: 10px; color: #e0e0e0;">Manish</td> <td style="padding: 10px; color: #e0e0e0;">82</td> <td style="padding: 10px; color: #e0e0e0;">Mumbai</td> </tr> </table> </div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #f093fb;"> <h4 style="color: #ffffff; margin-top: 0;">✨ Benefits of Custom Index</h4> <ul style="color: #e0e0e0; margin: 10px 0 0 20px; line-height: 1.8;"> <li><strong>Meaningful Labels:</strong> Use descriptive names instead of numeric indices</li> <li><strong>Easier Access:</strong> Reference rows by name: <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">data.loc['first']</code></li> <li><strong>Better Readability:</strong> Makes data exploration more intuitive</li> <li><strong>Date/Time Support:</strong> Can use datetime objects as index for time-series data</li> </ul> </div>

<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #38ef7d;"> <p style="color: #1e1e1e; margin: 0; font-size: 14px;"> 💡 <strong>Pro Tip:</strong> You can also set index during DataFrame creation: <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">pd.DataFrame(data, index=['first', 'second'])</code> or when reading CSV: <code style="background: #1e1e1e; padding: 3px 8px; border-radius: 4px; color: #feca57;">pd.read_csv('file.csv', index_col='column_name')</code> </p> </div>
