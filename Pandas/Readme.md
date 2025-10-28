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
<pre> name makrs city 0 Anish 92 Delhi 1 Manish 82 Mumbai </pre>

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
