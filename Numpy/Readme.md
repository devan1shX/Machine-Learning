# 🚀 NumPy Complete Guide - Phase 1 & 2

> A comprehensive, detailed guide to NumPy fundamentals and intermediate operations with extensive examples

---

## 📋 Table of Contents

- [Data Structures Overview](#data-structures-overview)
- [Phase 1: Foundation](#phase-1-foundation)
  - [Creating Arrays](#creating-arrays)
  - [List vs NumPy Array](#list-vs-numpy-array)
  - [Creating Arrays from Scratch](#creating-arrays-from-scratch)
  - [Understanding Dimensions](#understanding-dimensions)
  - [Array Properties](#array-properties)
  - [Array Reshaping](#array-reshaping)
- [Phase 2: Intermediate Operations](#phase-2-intermediate-operations)
  - [Indexing and Slicing](#indexing-and-slicing)
  - [Sorting Arrays](#sorting-arrays)
  - [Filtering and Conditionals](#filtering-and-conditionals)
  - [Adding and Removing Data](#adding-and-removing-data)

---

## 📊 Data Structures Overview

NumPy works with three main data structures:

### 📏 Vector (1D Array)
```python
[1, 2, 3, 4, 5, 6]
```
A **one-dimensional** array of numbers

### 📐 Matrix (2D Array)
```python
[[1, 2],
 [3, 4]]
```
A **two-dimensional** array with rows and columns

### 🧊 Tensors (3D+ Arrays)
```python
[[[1, 2], [3, 4]],
 [[5, 6], [7, 8]]]
```
**Multi-dimensional** arrays (3D and beyond)

### 💡 Quick Reference

| Type | Dimensions | Example |
|------|------------|---------|
| **Vector** | 1D | `[1, 2, 3, 4, 5, 6]` |
| **Matrix** | 2D | `[[1, 2], [3, 4]]` |
| **Tensors** | 3D+ | Multi-dimensional arrays |

---

## 📦 PHASE 1: Foundation

> **NumPy Arrays and Basics**

### Getting Started

```python
import numpy as np
```

---

## 🎯 Creating Arrays

### From Python Lists

Convert a Python list to a NumPy array:

#### Example 1: 1D Array
```python
my_list = [1, 2, 3, 4, 5]
arr_1d = np.array(my_list)
print(arr_1d)
# Output: array([1, 2, 3, 4, 5])
```

**Or directly:**
```python
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d)
# Output: array([1, 2, 3, 4, 5])
```

#### Example 2: 2D Array
```python
my_list_2d = [[1, 2], [3, 4]]
arr_2d = np.array(my_list_2d)
print(arr_2d)
# Output: array([[1, 2],
#                [3, 4]])
```

**Or directly:**
```python
arr_2d = np.array([[1, 2], [3, 4]])
print(arr_2d)
# Output: array([[1, 2],
#                [3, 4]])
```

#### Example 3: Creating from nested lists
```python
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr_3x3 = np.array(nested_list)
print(arr_3x3)
# Output: array([[1, 2, 3],
#                [4, 5, 6],
#                [7, 8, 9]])
```

**Key Point:** `np.array()` transforms Python lists into powerful NumPy arrays that support vectorized operations

---

## ⚖️ List vs NumPy Array

### Comparison Table

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| **Speed** | Slower ⏱️ | ⚡ Faster (50-100x) |
| **Memory** | More memory usage | 💾 Compact & efficient |
| **Operations** | Limited | Mathematical operations |
| **Type** | Mixed types allowed | Homogeneous (same type) |
| **Syntax** | Basic operations | Vectorized operations |

### 🔄 Behavior Comparison

#### Python List - Concatenation
```python
py_list = [1, 2, 3]
print(py_list)
# Output: [1, 2, 3]

py_list *= 2
print(py_list)
# Output: [1, 2, 3, 1, 2, 3]
```
**Explanation:** Lists **repeat** elements when multiplied. The `*` operator concatenates the list to itself.

#### NumPy Array - Element-wise Operation
```python
np_array = np.array([1, 2, 3])
print(np_array)
# Output: [1 2 3]

np_array *= 2
print(np_array)
# Output: [2 4 6]
```
**Explanation:** Arrays perform **mathematical operations** on each element. Each element is multiplied by 2.

### 📊 More Examples

#### Addition with Python Lists
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
result = list1 + list2
print(result)
# Output: [1, 2, 3, 4, 5, 6]
```
**Note:** Lists are concatenated, not added element-wise

#### Addition with NumPy Arrays
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = arr1 + arr2
print(result)
# Output: [5 7 9]
```
**Note:** Arrays perform element-wise addition: `[1+4, 2+5, 3+6] = [5, 7, 9]`

### ⏱️ Performance Comparison

```python
import time

# List operation
start = time.time()
py_list = list(range(1000000))
result = [x * 2 for x in py_list]
list_time = time.time() - start
print(f"List time: {list_time:.4f}s")

# NumPy operation
start = time.time()
np_array = np.arange(1000000)
result = np_array * 2
numpy_time = time.time() - start
print(f"NumPy time: {numpy_time:.4f}s")

print(f"NumPy is {list_time/numpy_time:.2f}x faster!")
```

**Example Output:**
```
List time: 0.0823s
NumPy time: 0.0012s
NumPy is 68.58x faster!
```

**💡 NumPy is significantly faster for large-scale numerical operations!**

---

## 🔨 Creating Arrays from Scratch

### `np.zeros()` - Array of Zeros

#### Example 1: 1D Array of Zeros
```python
zeros_1d = np.zeros(5)
print(zeros_1d)
# Output: [0. 0. 0. 0. 0.]
```
Creates a **1D array** with 5 zeros

#### Example 2: 2D Array of Zeros
```python
zeros_2d = np.zeros((3, 4))
print(zeros_2d)
# Output: [[0. 0. 0. 0.]
#          [0. 0. 0. 0.]
#          [0. 0. 0. 0.]]
```
Creates a **3×4 array** (3 rows, 4 columns) filled with zeros

#### Example 3: 3D Array of Zeros
```python
zeros_3d = np.zeros((2, 3, 2))
print(zeros_3d)
# Output: [[[0. 0.]
#           [0. 0.]
#           [0. 0.]]
#
#          [[0. 0.]
#           [0. 0.]
#           [0. 0.]]]
```
Creates a **2×3×2 tensor** filled with zeros

**Note:** By default, `np.zeros()` creates arrays with `float64` data type

---

### `np.ones()` - Array of Ones

#### Example 1: 1D Array of Ones
```python
ones_1d = np.ones(4)
print(ones_1d)
# Output: [1. 1. 1. 1.]
```
Creates a **1D array** with 4 ones

#### Example 2: 2D Array of Ones
```python
ones_2d = np.ones((2, 3))
print(ones_2d)
# Output: [[1. 1. 1.]
#          [1. 1. 1.]]
```
Creates a **2×3 array** filled with ones

#### Example 3: Specifying Data Type
```python
ones_int = np.ones((2, 2), dtype=int)
print(ones_int)
# Output: [[1 1]
#          [1 1]]
```
Creates a **2×2 array** of integer ones (not float)

---

### `np.full()` - Array with Custom Value

#### Example 1: Fill with Number 7
```python
full_7 = np.full((2, 2), 7)
print(full_7)
# Output: [[7 7]
#          [7 7]]
```
Creates a **2×2 array** filled with **7**

#### Example 2: Fill with Different Value
```python
full_42 = np.full((3, 3), 42)
print(full_42)
# Output: [[42 42 42]
#          [42 42 42]
#          [42 42 42]]
```
Creates a **3×3 array** filled with **42**

#### Example 3: Fill with Float Value
```python
full_pi = np.full((2, 4), 3.14)
print(full_pi)
# Output: [[3.14 3.14 3.14 3.14]
#          [3.14 3.14 3.14 3.14]]
```
Creates a **2×4 array** filled with **3.14**

---

### `np.random.random()` - Random Values

#### Example 1: 1D Random Array
```python
random_1d = np.random.random(5)
print(random_1d)
# Output: [0.234 0.876 0.123 0.987 0.456]  (values will vary)
```
Creates a **1D array** with 5 random values between **0 and 1**

#### Example 2: 2D Random Array
```python
random_2d = np.random.random((2, 3))
print(random_2d)
# Output: [[0.234 0.876 0.123]
#          [0.987 0.456 0.789]]  (values will vary)
```
Creates a **2×3 array** with random values between **0 and 1**

#### Example 3: Random Integers
```python
random_int = np.random.randint(1, 10, size=(3, 3))
print(random_int)
# Output: [[3 7 2]
#          [8 1 5]
#          [4 9 6]]  (values will vary)
```
Creates a **3×3 array** with random integers between **1 and 9**

**Note:** Each time you run these, you'll get different random values!

---

### `np.arange()` - Sequence with Steps

#### Example 1: Simple Range
```python
sequence_1 = np.arange(10)
print(sequence_1)
# Output: [0 1 2 3 4 5 6 7 8 9]
```
Creates a sequence from **0 to 9** (10 not included)

#### Example 2: With Start and Stop
```python
sequence_2 = np.arange(5, 15)
print(sequence_2)
# Output: [5 6 7 8 9 10 11 12 13 14]
```
Creates a sequence from **5 to 14**

#### Example 3: With Step
```python
sequence_3 = np.arange(0, 11, 3)
print(sequence_3)
# Output: [0 3 6 9]
```
Creates a sequence from **0 to 10** with a **step of 3**

#### Example 4: Float Steps
```python
sequence_4 = np.arange(0, 1, 0.2)
print(sequence_4)
# Output: [0.  0.2 0.4 0.6 0.8]
```
Creates a sequence with **float steps**

#### Example 5: Negative Steps
```python
sequence_5 = np.arange(10, 0, -2)
print(sequence_5)
# Output: [10  8  6  4  2]
```
Creates a **descending** sequence

**Syntax:** `np.arange(start, stop, step)`
- `start`: Starting value (inclusive)
- `stop`: Ending value (exclusive)
- `step`: Increment value (default is 1)

---

### `np.linspace()` - Evenly Spaced Values

#### Example 1: Basic Usage
```python
linspace_1 = np.linspace(0, 10, 5)
print(linspace_1)
# Output: [ 0.   2.5  5.   7.5 10. ]
```
Creates **5 evenly spaced values** between **0 and 10** (inclusive)

#### Example 2: More Points
```python
linspace_2 = np.linspace(0, 1, 11)
print(linspace_2)
# Output: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
```
Creates **11 evenly spaced values** between **0 and 1**

**Difference from `arange()`:**
- `arange()`: You specify the **step size**
- `linspace()`: You specify the **number of points**

---

## 🔢 Understanding Dimensions

### 📏 Vector (1D Array)

#### Example 1: Simple Vector
```python
vector_1 = np.array([1, 2, 3])
print(vector_1)
# Output: [1 2 3]
print(f"Shape: {vector_1.shape}")
# Output: Shape: (3,)
```
A **one-dimensional** array with 3 elements

#### Example 2: Larger Vector
```python
vector_2 = np.array([10, 20, 30, 40, 50])
print(vector_2)
# Output: [10 20 30 40 50]
print(f"Shape: {vector_2.shape}")
# Output: Shape: (5,)
```

**Key Points:**
- Vectors have only **one dimension**
- Shape is represented as `(n,)` where n is the number of elements
- Think of it as a **single row** or **single column**

---

### 📐 Matrix (2D Array)

#### Example 1: 2×2 Matrix
```python
matrix_1 = np.array([[1, 2], 
                     [3, 4]])
print(matrix_1)
# Output: [[1 2]
#          [3 4]]
print(f"Shape: {matrix_1.shape}")
# Output: Shape: (2, 2)
```
A **2×2 matrix** (2 rows, 2 columns)

#### Example 2: 3×3 Matrix
```python
matrix_2 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
print(matrix_2)
# Output: [[1 2 3]
#          [4 5 6]
#          [7 8 9]]
print(f"Shape: {matrix_2.shape}")
# Output: Shape: (3, 3)
```

#### Example 3: Non-Square Matrix
```python
matrix_3 = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8]])
print(matrix_3)
# Output: [[1 2 3 4]
#          [5 6 7 8]]
print(f"Shape: {matrix_3.shape}")
# Output: Shape: (2, 4)
```
A **2×4 matrix** (2 rows, 4 columns)

**Key Points:**
- Matrices have **two dimensions** (rows and columns)
- Shape is represented as `(rows, columns)`
- Think of it as a **table** or **grid**

---

### 🧊 Tensor (3D+ Array)

#### Example 1: 3D Tensor
```python
tensor_1 = np.array([[[1, 2], [3, 4]], 
                     [[5, 6], [7, 8]]])
print(tensor_1)
# Output: [[[1 2]
#           [3 4]]
#
#          [[5 6]
#           [6 8]]]
print(f"Shape: {tensor_1.shape}")
# Output: Shape: (2, 2, 2)
```

**Understanding the shape (2, 2, 2):**
- **First dimension (2)**: 2 "blocks" or "matrices"
- **Second dimension (2)**: 2 rows in each block
- **Third dimension (2)**: 2 columns in each row

#### Example 2: Different 3D Tensor
```python
tensor_2 = np.array([[[1, 2, 3],
                      [4, 5, 6]],
                     
                     [[7, 8, 9],
                      [10, 11, 12]]])
print(tensor_2)
print(f"Shape: {tensor_2.shape}")
# Output: Shape: (2, 2, 3)
```

**Understanding the shape (2, 2, 3):**
- **2 blocks** of data
- Each block has **2 rows**
- Each row has **3 columns**

#### Visual Representation
```
Think of a 3D array as a stack of matrices:

Block 0:          Block 1:
[[1 2 3]          [[7  8  9]
 [4 5 6]]          [10 11 12]]
```

**Key Points:**
- Tensors have **three or more dimensions**
- Common in deep learning (images, videos, etc.)
- Shape is `(depth, rows, columns)` for 3D arrays

---

### 💡 Dimension Quick Reference

| Type | Dimensions | Shape Example | Visual |
|------|------------|---------------|--------|
| **Vector** | 1D | `(5,)` | `[1 2 3 4 5]` |
| **Matrix** | 2D | `(3, 4)` | Table with 3 rows, 4 columns |
| **Tensor** | 3D+ | `(2, 3, 4)` | Stack of 2 matrices (3×4 each) |

---

## 📊 Array Properties

### Understanding Core Properties

```python
arr = np.array([1, 2, 3])
```

---

### 📷 `shape` - Dimensions of the Array

The `shape` attribute tells you the **size along each dimension**.

#### Example 1: 1D Array
```python
arr_1d = np.array([1, 2, 3])
print(arr_1d.shape)
# Output: (3,)
```
**Explanation:** 
- Shape `(3,)` means it's a 1D array with **3 elements**
- The comma after 3 indicates it's a tuple with one element

#### Example 2: 2D Array
```python
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6]])
print(arr_2d.shape)
# Output: (2, 3)
```
**Explanation:**
- Shape `(2, 3)` means **2 rows** and **3 columns**
- First number = rows, Second number = columns

#### Example 3: 3D Array
```python
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
print(arr_3d.shape)
# Output: (2, 2, 2)
```
**Explanation:**
- Shape `(2, 2, 2)` means:
  - **2** matrices/blocks
  - **2** rows in each matrix
  - **2** columns in each row

#### Example 4: Different Shapes
```python
arr_a = np.array([1, 2, 3, 4, 5, 6])
arr_b = np.array([[1, 2, 3], [4, 5, 6]])
arr_c = np.array([[1], [2], [3], [4]])

print(f"arr_a shape: {arr_a.shape}")  # (6,)
print(f"arr_b shape: {arr_b.shape}")  # (2, 3)
print(f"arr_c shape: {arr_c.shape}")  # (4, 1)
```

**Visual Comparison:**
```
arr_a: [1 2 3 4 5 6]                    → (6,)
arr_b: [[1 2 3]                         → (2, 3)
        [4 5 6]]
arr_c: [[1]                             → (4, 1)
        [2]
        [3]
        [4]]
```

---

### 📷 `ndim` - Number of Dimensions

The `ndim` attribute tells you **how many axes** (dimensions) the array has.

#### Example 1: 1D Array
```python
arr_1d = np.array([1, 2, 3])
print(arr_1d.ndim)
# Output: 1
```
**Explanation:** It has **1 dimension** (a vector)

#### Example 2: 2D Array
```python
arr_2d = np.array([[1, 2], [3, 4]])
print(arr_2d.ndim)
# Output: 2
```
**Explanation:** It has **2 dimensions** (a matrix with rows and columns)

#### Example 3: 3D Array
```python
arr_3d = np.array([[[1, 2]], [[3, 4]]])
print(arr_3d.ndim)
# Output: 3
```
**Explanation:** It has **3 dimensions** (a tensor)

#### Example 4: Scalar (0D)
```python
arr_0d = np.array(42)
print(arr_0d.ndim)
# Output: 0
```
**Explanation:** A single number has **0 dimensions** (it's a scalar)

#### Comprehensive Example
```python
scalar = np.array(5)
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])
tensor = np.array([[[1, 2]], [[3, 4]]])

print(f"Scalar ndim: {scalar.ndim}")   # 0
print(f"Vector ndim: {vector.ndim}")   # 1
print(f"Matrix ndim: {matrix.ndim}")   # 2
print(f"Tensor ndim: {tensor.ndim}")   # 3
```

**Visual Understanding:**
```
0D (Scalar): 5
1D (Vector): [1 2 3]
2D (Matrix): [[1 2]
              [3 4]]
3D (Tensor): [[[1 2]]
              [[3 4]]]
```

---

### 📷 `size` - Total Number of Elements

The `size` attribute tells you the **total count** of all elements in the array.

#### Example 1: 1D Array
```python
arr_1d = np.array([1, 2, 3])
print(arr_1d.size)
# Output: 3
```
**Explanation:** There are **3 elements** total

#### Example 2: 2D Array
```python
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6]])
print(arr_2d.size)
# Output: 6
```
**Explanation:** 
- Shape is `(2, 3)` → 2 rows × 3 columns
- Total elements: 2 × 3 = **6 elements**

#### Example 3: 3D Array
```python
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
print(arr_3d.size)
# Output: 8
```
**Explanation:**
- Shape is `(2, 2, 2)` → 2 × 2 × 2
- Total elements: 2 × 2 × 2 = **8 elements**

#### Example 4: Calculating Size from Shape
```python
arr = np.zeros((3, 4, 5))
print(f"Shape: {arr.shape}")     # (3, 4, 5)
print(f"Size: {arr.size}")       # 60

# Verify: 3 × 4 × 5 = 60
print(f"Calculated: {3 * 4 * 5}") # 60
```

**Formula:** `size = product of all dimensions`

---

### 📷 `dtype` - Data Type of Elements

The `dtype` attribute tells you the **data type** of elements in the array.

#### Example 1: Integer Array
```python
arr_int = np.array([1, 2, 3])
print(arr_int.dtype)
# Output: dtype('int64')
```
**Explanation:** Elements are **64-bit integers**

#### Example 2: Float Array
```python
arr_float = np.array([1.5, 2.5, 3.5])
print(arr_float.dtype)
# Output: dtype('float64')
```
**Explanation:** Elements are **64-bit floating-point numbers**

#### Example 3: Explicitly Setting dtype
```python
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
print(arr_int32.dtype)
# Output: dtype('int32')
```

#### Example 4: String Array
```python
arr_str = np.array(['a', 'b', 'c'])
print(arr_str.dtype)
# Output: dtype('<U1')
```
**Explanation:** Unicode strings with max length 1

#### Common Data Types
```python
# Integer types
int8_arr = np.array([1, 2], dtype=np.int8)      # 8-bit integer
int16_arr = np.array([1, 2], dtype=np.int16)    # 16-bit integer
int32_arr = np.array([1, 2], dtype=np.int32)    # 32-bit integer
int64_arr = np.array([1, 2], dtype=np.int64)    # 64-bit integer

# Float types
float32_arr = np.array([1.0, 2.0], dtype=np.float32)  # 32-bit float
float64_arr = np.array([1.0, 2.0], dtype=np.float64)  # 64-bit float

# Boolean
bool_arr = np.array([True, False], dtype=bool)

print(f"int8: {int8_arr.dtype}")
print(f"float32: {float32_arr.dtype}")
print(f"bool: {bool_arr.dtype}")
```

---

### 🧪 Type Conversion Examples

#### Mixed Integer and Boolean

```python
arr_2 = np.array([1, 2, True])
print(arr_2)
# Output: [1 2 1]

print(arr_2.dtype)
# Output: dtype('int64')
```

**Explanation:**
- NumPy converts `True` to `1`
- All elements become integers
- Boolean `True` = 1, `False` = 0

#### Mixed Float, Integer, and Boolean

```python
arr_3 = np.array([1, 4.5, True])
print(arr_3)
# Output: [1.  4.5 1. ]

print(arr_3.dtype)
# Output: dtype('float64')
```

**Explanation:**
- Integer `1` converts to `1.0`
- Boolean `True` converts to `1.0`
- All become floats (highest precision type)

#### Mixed Integer and Float

```python
arr_4 = np.array([1, 2, 3.5])
print(arr_4)
# Output: [1.  2.  3.5]

print(arr_4.dtype)
# Output: dtype('float64')
```

**Explanation:**
- Integers convert to floats
- Result: all elements are `float64`

#### String Takes Priority

```python
arr_5 = np.array([1, 2, 'hello'])
print(arr_5)
# Output: ['1' '2' 'hello']

print(arr_5.dtype)
# Output: dtype('<U21')
```

**Explanation:**
- All elements convert to strings
- `1` becomes `'1'`, `2` becomes `'2'`
- String type has highest priority

---

### 📋 Complete Property Example

```python
# Create a 2×3 matrix
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

print("Array:")
print(matrix)
print()

# All properties
print(f"shape:  {matrix.shape}")   # (2, 3)    → 2 rows, 3 columns
print(f"ndim:   {matrix.ndim}")    # 2         → 2 dimensions (matrix)
print(f"size:   {matrix.size}")    # 6         → 6 total elements
print(f"dtype:  {matrix.dtype}")   # int64     → 64-bit integers
```

**Output:**
```
Array:
[[1 2 3]
 [4 5 6]]

shape:  (2, 3)
ndim:   2
size:   6
dtype:  dtype('int64')
```

---

### 💡 Properties Reference Table

| Property | What it tells you | 1D Example | 2D Example | 3D Example |
|----------|-------------------|------------|------------|------------|
| **shape** | Size along each dimension | `(5,)` | `(2, 3)` | `(2, 2, 2)` |
| **ndim** | Number of dimensions | `1` | `2` | `3` |
| **size** | Total element count | `5` | `6` | `8` |
| **dtype** | Data type of elements | `int64` | `float64` | `int32` |

---

### 🎯 Key Takeaways

1. **shape** shows the structure: `(rows, columns)` for 2D
2. **ndim** counts the axes: 1D → 1, 2D → 2, 3D → 3
3. **size** multiplies all dimensions: `shape (2,3)` → `size 6`
4. **dtype** maintains consistency: NumPy converts to the most general type

> **Rule:** NumPy automatically converts mixed types to maintain array homogeneity. The conversion hierarchy is: **int → float → string**

---

## 🔄 Array Reshaping

Reshaping allows you to change the dimensions of an array without changing its data.

### Creating the Array

```python
arr = np.arange(12)
print(arr)
# Output: [0 1 2 3 4 5 6 7 8 9 10 11]
print(f"Original shape: {arr.shape}")
# Output: Original shape: (12,)
```
Creates a 1D array with values from **0 to 11**

---

### 📐 `reshape()` - Change Array Dimensions

The `reshape()` method transforms an array into a new shape without changing the data.

#### Example 1: 1D to 2D
```python
arr = np.arange(12)
reshaped = arr.reshape((3, 4))
print(reshaped)
# Output: [[ 0  1  2  3]
#          [ 4  5  6  7]
#          [ 8  9 10 11]]
print(f"New shape: {reshaped.shape}")
# Output: New shape: (3, 4)
```
**Explanation:** Converts `(12,)` → `(3, 4)` (3 rows × 4 columns)

#### Example 2: Different 2D Shape
```python
arr = np.arange(12)
reshaped = arr.reshape((4, 3))
print(reshaped)
# Output: [[ 0  1  2]
#          [ 3  4  5]
#          [ 6  7  8]
#          [ 9 10 11]]
```
**Explanation:** Same data, but now **4 rows × 3 columns**

#### Example 3: 1D to 3D
```python
arr = np.arange(12)
reshaped = arr.reshape((2, 2, 3))
print(reshaped)
# Output: [[[ 0  1  2]
#           [ 3  4  5]]
#
#          [[ 6  7  8]
#           [ 9 10 11]]]
print(f"Shape: {reshaped.shape}")
# Output: Shape: (2, 2, 3)
```
**Explanation:** Creates a 3D array with 2 blocks, each having 2 rows and 3 columns

#### Example 4: Using -1 (Auto-calculate)
```python
arr = np.arange(12)
reshaped = arr.reshape((3, -1))
print(reshaped)
# Output: [[ 0  1  2  3]
#          [ 4  5  6  7]
#          [ 8  9 10 11]]
print(f"Shape: {reshaped.shape}")
# Output: Shape: (3, 4)
```
**Explanation:** `-1` means "figure out this dimension automatically"
- We want 3 rows
- NumPy calculates: 12 elements ÷ 3 rows = 4 columns

#### Example 5: Another -1 Usage
```python
arr = np.arange(12)
reshaped = arr.reshape((-1, 6))
print(reshaped)
# Output: [[ 0  1  2  3  4  5]
#          [ 6  7  8  9 10 11]]
print(f"Shape: {reshaped.shape}")
# Output: Shape: (2, 6)
```
**Explanation:** We want 6 columns, NumPy calculates 2 rows

**Important Rule:** The total number of elements must remain the same!
```python
# Valid: 12 elements → (3, 4) = 12 ✓
# Valid: 12 elements → (2, 6) = 12 ✓
# Invalid: 12 elements → (3, 5) = 15 ✗ (will raise error)
```

---

### 📏 `flatten()` - Convert to 1D (Copy)

The `flatten()` method converts a multi-dimensional array into a 1D array by creating a **new copy**.

#### Example 1: Flatten 2D Array
```python
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6]])
flattened = arr_2d.flatten()
print(flattened)
# Output: [1 2 3 4 5 6]
print(f"Shape: {flattened.shape}")
# Output: Shape: (6,)
```

#### Example 2: Flatten 3D Array
```python
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
flattened = arr_3d.flatten()
print(flattened)
# Output: [1 2 3 4 5 6 7 8]
```

#### Example 3: Modifying Flattened Array (Creates Copy)
```python
original = np.array([[1, 2], [3, 4]])
flattened = original.flatten()

# Modify the flattened array
flattened[0] = 999

print("Original array:")
print(original)
# Output: [[1 2]
#          [3 4]]

print("Flattened array:")
print(flattened)
# Output: [999   2   3   4]
```
**Key Point:** Changes to `flattened` **don't affect** `original` because `flatten()` creates a **copy**

---

### 📏 `ravel()` - Convert to 1D (View)

The `ravel()` method also converts to 1D but returns a **view** (reference) when possible, not a copy.

#### Example 1: Ravel 2D Array
```python
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6]])
raveled = arr_2d.ravel()
print(raveled)
# Output: [1 2 3 4 5 6]
```

#### Example 2: Modifying Raveled Array (Shares Memory)
```python
original = np.array([[1, 2], [3, 4]])
raveled = original.ravel()

# Modify the raveled array
raveled[0] = 999

print("Original array:")
print(original)
# Output: [[999   2]
#          [  3   4]]

print("Raveled array:")
print(raveled)
# Output: [999   2   3   4]
```
**Key Point:** Changes to `raveled` **affect** `original` because `ravel()` creates a **view** that shares memory

#### Example 3: Performance Comparison
```python
import time

large_array = np.arange(1000000).reshape((1000, 1000))

# Time flatten (creates copy)
start = time.time()
flat = large_array.flatten()
flatten_time = time.time() - start

# Time ravel (creates view)
start = time.time()
rav = large_array.ravel()
ravel_time = time.time() - start

print(f"flatten() time: {flatten_time:.6f}s")
print(f"ravel() time: {ravel_time:.6f}s")
print(f"ravel() is {flatten_time/ravel_time:.2f}x faster")
```

**Expected Output:**
```
flatten() time: 0.002341s
ravel() time: 0.000012s
ravel() is 195.08x faster
```

---

### 🔃 `T` - Transpose (Flip Rows ↔ Columns)

The `T` attribute swaps rows and columns (transposes the array).

#### Example 1: Transpose 2D Array
```python
arr = np.array([[1, 2, 3], 
                [4, 5, 6]])
print("Original:")
print(arr)
# Output: [[1 2 3]
#          [4 5 6]]
print(f"Shape: {arr.shape}")  # (2, 3)

transposed = arr.T
print("\nTransposed:")
print(transposed)
# Output: [[1 4]
#          [2 5]
#          [3 6]]
print(f"Shape: {transposed.shape}")  # (3, 2)
```

**Visual Understanding:**
```
Original (2×3):          Transposed (3×2):
[[1 2 3]                 [[1 4]
 [4 5 6]]                 [2 5]
                          [3 6]]
```

#### Example 2: Square Matrix Transpose
```python
square = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("Original:")
print(square)
# Output: [[1 2 3]
#          [4 5 6]
#          [7 8 9]]

transposed = square.T
print("\nTransposed:")
print(transposed)
# Output: [[1 4 7]
#          [2 5 8]
#          [3 6 9]]
```

#### Example 3: 1D Array Transpose
```python
arr_1d = np.array([1, 2, 3, 4])
print("Original:")
print(arr_1d)
# Output: [1 2 3 4]

transposed = arr_1d.T
print("Transposed:")
print(transposed)
# Output: [1 2 3 4]  (no change for 1D)
```
**Note:** Transposing a 1D array has **no effect**

#### Example 4: Using transpose() Method
```python
arr = np.array([[1, 2], [3, 4]])

# Method 1: Using .T
trans1 = arr.T

# Method 2: Using transpose()
trans2 = arr.transpose()

print(np.array_equal(trans1, trans2))
# Output: True
```

#### Example 5: 3D Array Transpose
```python
arr_3d = np.arange(24).reshape((2, 3, 4))
print(f"Original shape: {arr_3d.shape}")
# Output: Original shape: (2, 3, 4)

transposed = arr_3d.T
print(f"Transposed shape: {transposed.shape}")
# Output: Transposed shape: (4, 3, 2)
```
**Note:** For 3D arrays, transpose reverses all dimensions

---

### 💡 Reshaping Methods Comparison

| Method | What it does | Creates Copy? | Memory | Speed | Use Case |
|--------|--------------|---------------|---------|-------|----------|
| **reshape()** | Changes dimensions | No (view) | Shared | Fast | Change shape without copying |
| **flatten()** | Converts to 1D | ✅ Yes | New copy | Slower | When you need independent copy |
| **ravel()** | Converts to 1D | ❌ No (view) | Shared | Faster | When you can share memory |
| **T** | Swaps rows/columns | No (view) | Shared | Fast | Matrix operations |

### 🎯 Key Differences: `flatten()` vs `ravel()`

```python
original = np.array([[1, 2], [3, 4]])

# flatten() - creates copy
flat = original.flatten()
flat[0] = 999
print(original[0, 0])  # Output: 1 (unchanged)

# ravel() - creates view
rav = original.ravel()
rav[0] = 999
print(original[0, 0])  # Output: 999 (changed!)
```

**Summary:**
- Use `flatten()` when you need an **independent copy**
- Use `ravel()` when you want **better performance** and can share memory
- Use `T` for **matrix transpose operations**
- Use `reshape()` with `-1` for **automatic dimension calculation**

---

## 📦 PHASE 2: Intermediate Operations

> **Advanced Array Manipulation & Techniques**

---

## 🎯 Indexing and Slicing

Indexing and slicing allow you to access and manipulate specific elements or portions of arrays.

### 1D Array Slicing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr)
# Output: [1 2 3 4 5 6 7 8 9]
```

---

#### 📌 Basic Indexing

##### Example 1: Single Element Access
```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])   # Output: 10  (first element)
print(arr[2])   # Output: 30  (third element)
print(arr[-1])  # Output: 50  (last element)
print(arr[-2])  # Output: 40  (second-to-last)
```

**Visual:**
```
Array:  [10  20  30  40  50]
Index:   0   1   2   3   4
Neg:    -5  -4  -3  -2  -1
```

##### Example 2: Modifying Elements
```python
arr = np.array([1, 2, 3, 4, 5])
arr[0] = 100
arr[-1] = 500
print(arr)
# Output: [100   2   3   4 500]
```

---

#### 📌 Basic Slicing

##### Example 1: Slice with Start and Stop
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[2:7])
# Output: [3 4 5 6 7]
```
**Explanation:** 
- Start at index **2** (value 3)
- Stop before index **7** (stop is exclusive)
- Gets indices: 2, 3, 4, 5, 6

##### Example 2: From Start
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[:5])
# Output: [1 2 3 4 5]
```
**Explanation:** From beginning to index 5 (exclusive)

##### Example 3: To End
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[5:])
# Output: [6 7 8 9]
```
**Explanation:** From index 5 to the end

##### Example 4: Entire Array
```python
arr = np.array([1, 2, 3, 4, 5])

print(arr[:])
# Output: [1 2 3 4 5]
```
**Explanation:** Copy of the entire array

---

#### 🪜 Slicing with Steps

##### Example 1: Every 2nd Element
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[0:10:2])
# Output: [1 3 5 7 9]
```
**Explanation:** Start at 0, stop at 10, step by 2

##### Example 2: Every 3rd Element
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[::3])
# Output: [1 4 7]
```
**Explanation:** Start to end, step by 3

##### Example 3: Skip First, Take Every 2nd
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[1::2])
# Output: [2 4 6 8]
```
**Explanation:** Start at index 1, take every 2nd element

---

#### 🔄 Negative Indexing and Reverse

##### Example 1: Complete Reverse
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[::-1])
# Output: [9 8 7 6 5 4 3 2 1]
```
**Explanation:** Step of `-1` reverses the array

##### Example 2: Reverse Every 2nd Element
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[::-2])
# Output: [9 7 5 3 1]
```
**Explanation:** Start from end, step backwards by 2

##### Example 3: Negative Indices
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[-5:-1])
# Output: [5 6 7 8]
```
**Explanation:** From 5th-last to 1st-last (exclusive)

##### Example 4: Mixed Negative and Positive
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[2:-2])
# Output: [3 4 5 6 7]
```
**Explanation:** From index 2 to 2nd-last (exclusive)

---

### 📐 2D Array Indexing and Slicing

```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(arr_2d)
# Output: [[1 2 3]
#          [4 5 6]
#          [7 8 9]]
```

---

#### 🎯 Accessing Specific Elements

##### Example 1: Single Element (NumPy Style - Preferred)
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[0, 2])  # Output: 3
print(arr_2d[1, 1])  # Output: 5
print(arr_2d[2, 0])  # Output: 7
```
**Explanation:** Format is `[row, column]`
- `[0, 2]` = row 0, column 2 = value 3
- `[1, 1]` = row 1, column 1 = value 5

##### Example 2: Single Element (List Style)
```python
print(arr_2d[0][2])  # Output: 3
print(arr_2d[1][1])  # Output: 5
```
**Note:** Works but less efficient than NumPy style

##### Example 3: Negative Indexing
```python
print(arr_2d[-1, -1])  # Output: 9 (last row, last column)
print(arr_2d[-2, -3])  # Output: 4 (2nd-last row, 3rd-last column)
```

##### Example 4: Modifying Elements
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

arr_2d[0, 0] = 100
arr_2d[1, 2] = 600
print(arr_2d)
# Output: [[100   2   3]
#          [  4   5 600]
#          [  7   8   9]]
```

---

#### 📊 Accessing Entire Rows

##### Example 1: Single Row
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[0])     # Output: [1 2 3] (first row)
print(arr_2d[1])     # Output: [4 5 6] (second row)
print(arr_2d[-1])    # Output: [7 8 9] (last row)
```

##### Example 2: Multiple Rows
```python
print(arr_2d[0:2])
# Output: [[1 2 3]
#          [4 5 6]]
```
**Explanation:** Rows 0 and 1 (2 is exclusive)

##### Example 3: Every Other Row
```python
arr_large = np.array([[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8]])

print(arr_large[::2])
# Output: [[1 2]
#          [5 6]]
```

---

#### 📊 Accessing Entire Columns

##### Example 1: Single Column
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[:, 0])  # Output: [1 4 7] (first column)
print(arr_2d[:, 1])  # Output: [2 5 8] (second column)
print(arr_2d[:, 2])  # Output: [3 6 9] (third column)
```
**Explanation:** `:` means "all rows", then specify column index

##### Example 2: Last Column
```python
print(arr_2d[:, -1])  # Output: [3 6 9]
```

##### Example 3: Multiple Columns
```python
print(arr_2d[:, 0:2])
# Output: [[1 2]
#          [4 5]
#          [7 8]]
```
**Explanation:** All rows, columns 0 and 1

---

#### 🎨 Advanced 2D Slicing

##### Example 1: Sub-matrix
```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(arr_2d[0:2, 1:3])
# Output: [[2 3]
#          [6 7]]
```
**Explanation:** 
- Rows: 0 to 1 (2 is exclusive)
- Columns: 1 to 2 (3 is exclusive)

##### Example 2: Every Other Element in Sub-matrix
```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

print(arr_2d[::2, ::2])
# Output: [[ 1  3]
#          [ 9 11]]
```
**Explanation:** Every 2nd row, every 2nd column

##### Example 3: Reverse Rows
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[::-1])
# Output: [[7 8 9]
#          [4 5 6]
#          [1 2 3]]
```

##### Example 4: Reverse Columns
```python
print(arr_2d[:, ::-1])
# Output: [[3 2 1]
#          [6 5 4]
#          [9 8 7]]
```

##### Example 5: Reverse Both
```python
print(arr_2d[::-1, ::-1])
# Output: [[9 8 7]
#          [6 5 4]
#          [3 2 1]]
```

---

### 💡 Slicing Syntax Summary

#### 1D Array
```python
arr[start:stop:step]
```
- `start`: Starting index (inclusive)
- `stop`: Ending index (exclusive)
- `step`: Increment (default 1)

#### 2D Array
```python
arr[row_start:row_stop:row_step, col_start:col_stop:col_step]
```

#### Common Patterns
```python
arr[:]          # All elements
arr[::2]        # Every 2nd element
arr[::-1]       # Reverse
arr[2:]         # From index 2 to end
arr[:5]         # From start to index 5
arr[2:8:2]      # From 2 to 8, step 2

# 2D
arr[0, :]       # First row
arr[:, 0]       # First column
arr[1:3, 2:4]   # Sub-matrix
```

---

## 🔃 Sorting Arrays

Sorting arranges array elements in ascending or descending order.

### 1D Array Sorting

#### Example 1: Basic Sorting
```python
unsorted_arr = np.array([3, 2, 6, 4, 3, 2, 6])
sorted_arr = np.sort(unsorted_arr)
print(sorted_arr)
# Output: [2 2 3 3 4 6 6]
```
**Explanation:** Sorts in **ascending order** by default

#### Example 2: Descending Order
```python
unsorted_arr = np.array([3, 2, 6, 4, 3, 2, 6])
sorted_desc = np.sort(unsorted_arr)[::-1]
print(sorted_desc)
# Output: [6 6 4 3 3 2 2]
```
**Explanation:** Sort ascending, then reverse

#### Example 3: In-place Sorting
```python
arr = np.array([5, 2, 8, 1, 9])
arr.sort()  # Sorts the array in-place
print(arr)
# Output: [1 2 5 8 9]
```
**Note:** `arr.sort()` modifies the original array

#### Example 4: np.sort() vs .sort()
```python
original = np.array([5, 2, 8, 1, 9])

# np.sort() - returns sorted copy
sorted_copy = np.sort(original)
print("Original:", original)    # [5 2 8 1 9] (unchanged)
print("Sorted:", sorted_copy)   # [1 2 5 8 9]

# .sort() - sorts in-place
original.sort()
print("Original:", original)    # [1 2 5 8 9] (changed!)
```

---

### 2D Array Sorting

```python
arr_2d_unsorted = np.array([[1, 3],
                            [4, 1],
                            [19, 11]])
print("Original:")
print(arr_2d_unsorted)
# Output: [[ 1  3]
#          [ 4  1]
#          [19 11]]
```

---

#### 📏 Column-wise Sorting (`axis=0`)

##### Example 1: Sort Down Columns
```python
arr_2d = np.array([[1, 3],
                   [4, 1],
                   [19, 11]])

sorted_col = np.sort(arr_2d, axis=0)
print(sorted_col)
# Output: [[ 1  1]
#          [ 4  3]
#          [19 11]]
```

**How it works - Column by Column:**
```
Column 0: [1, 4, 19] → sorted → [1, 4, 19]
Column 1: [3, 1, 11] → sorted → [1, 3, 11]

Result: [[ 1  1]
         [ 4  3]
         [19 11]]
```

##### Example 2: Larger Matrix
```python
arr = np.array([[5, 2, 8],
                [1, 9, 3],
                [7, 4, 6]])

sorted_col = np.sort(arr, axis=0)
print(sorted_col)
# Output: [[1 2 3]
#          [5 4 6]
#          [7 9 8]]
```

**Visual Explanation:**
```
Original:        Sort each column:
[[5 2 8]         [[1 2 3]
 [1 9 3]    →     [5 4 6]
 [7 4 6]]         [7 9 8]]

Col 0: [5,1,7] → [1,5,7]
Col 1: [2,9,4] → [2,4,9]
Col 2: [8,3,6] → [3,6,8]
```

---

#### 📏 Row-wise Sorting (`axis=1`)

##### Example 1: Sort Across Rows
```python
arr_2d = np.array([[1, 3],
                   [4, 1],
                   [19, 11]])

sorted_row = np.sort(arr_2d, axis=1)
print(sorted_row)
# Output: [[ 1  3]
#          [ 1  4]
#          [11 19]]
```

**How it works - Row by Row:**
```
Row 0: [1, 3]   → sorted → [1, 3]
Row 1: [4, 1]   → sorted → [1, 4]
Row 2: [19, 11] → sorted → [11, 19]

Result: [[ 1  3]
         [ 1  4]
         [11 19]]
```

##### Example 2: Larger Matrix
```python
arr = np.array([[5, 2, 8],
                [1, 9, 3],
                [7, 4, 6]])

sorted_row = np.sort(arr, axis=1)
print(sorted_row)
# Output: [[2 5 8]
#          [1 3 9]
#          [4 6 7]]
```

**Visual Explanation:**
```
Original:        Sort each row:
[[5 2 8]         [[2 5 8]
 [1 9 3]    →     [1 3 9]
 [7 4 6]]         [4 6 7]]

Row 0: [5,2,8] → [2,5,8]
Row 1: [1,9,3] → [1,3,9]
Row 2: [7,4,6] → [4,6,7]
```

---

#### 🔍 Understanding Axis

##### Comprehensive Example
```python
arr = np.array([[9, 2, 7],
                [3, 8, 1],
                [6, 4, 5]])

print("Original:")
print(arr)
print()

print("axis=0 (sort down columns):")
print(np.sort(arr, axis=0))
print()

print("axis=1 (sort across rows):")
print(np.sort(arr, axis=1))
print()

print("axis=None (flatten then sort):")
print(np.sort(arr, axis=None))
```

**Output:**
```
Original:
[[9 2 7]
 [3 8 1]
 [6 4 5]]

axis=0 (sort down columns):
[[3 2 1]
 [6 4 5]
 [9 8 7]]

axis=1 (sort across rows):
[[2 7 9]
 [1 3 8]
 [4 5 6]]

axis=None (flatten then sort):
[1 2 3 4 5 6 7 8 9]
```

---

### 💡 Axis Understanding Table

| Axis | Direction | What it does | Example Shape Change |
|------|-----------|--------------|---------------------|
| **axis=0** | ↓ Vertical | Sorts **down columns** | Shape stays same |
| **axis=1** | → Horizontal | Sorts **across rows** | Shape stays same |
| **axis=None** | — | Flattens then sorts | (3,3) → (9,) |

**Remember:** 
- `axis=0` operates **between rows** (vertically)
- `axis=1` operates **between columns** (horizontally)

---

### 📊 Advanced Sorting

#### Example 1: argsort() - Get Sorted Indices
```python
arr = np.array([3, 1, 4, 1, 5])
indices = np.argsort(arr)
print("Indices:", indices)
# Output: Indices: [1 3 0 2 4]

print("Sorted array:", arr[indices])
# Output: Sorted array: [1 1 3 4 5]
```
**Explanation:** `argsort()` returns the **indices** that would sort the array
- Index 1 has value 1 (smallest)
- Index 3 has value 1 (second smallest)
- Index 0 has value 3, etc.

#### Example 2: Sorting by Specific Column
```python
arr = np.array([[3, 1],
                [1, 4],
                [2, 2]])

# Sort rows by first column
indices = np.argsort(arr[:, 0])
sorted_arr = arr[indices]
print(sorted_arr)
# Output: [[1 4]
#          [2 2]
#          [3 1]]
```

---

## 🔍 Filtering and Conditionals

Filtering allows you to select elements based on conditions.

### Creating the Array

```python
num = np.arange(10)
print(num)
# Output: [0 1 2 3 4 5 6 7 8 9]
```

---

### 🎯 Method 1: Direct Boolean Filtering

#### Example 1: Even Numbers
```python
num = np.arange(10)
even_nums = num[num % 2 == 0]
print(even_nums)
# Output: [0 2 4 6 8]
```

**How it works:**
```
Step 1: num % 2 == 0 creates boolean array
[True, False, True, False, True, False, True, False, True, False]

Step 2: Use boolean array to filter
num[boolean_array] returns only True positions
Result: [0 2 4 6 8]
```

#### Example 2: Greater Than 5
```python
greater_5 = num[num > 5]
print(greater_5)
# Output: [6 7 8 9]
```

#### Example 3: Multiple Conditions (AND)
```python
# Numbers greater than 2 AND less than 7
filtered = num[(num > 2) & (num < 7)]
print(filtered)
# Output: [3 4 5 6]
```
**Note:** Use `&` for AND, `|` for OR, must use parentheses!

#### Example 4: Multiple Conditions (OR)
```python
# Numbers less than 3 OR greater than 7
filtered = num[(num < 3) | (num > 7)]
print(filtered)
# Output: [0 1 2 8 9]
```

#### Example 5: Not Equal
```python
# All numbers except 5
filtered = num[num != 5]
print(filtered)
# Output: [0 1 2 3 4 6 7 8 9]
```

---

### 🎭 Method 2: Using Masks

#### What is a Mask?

A mask is a **boolean array** that acts as a filter.

##### Example 1: Creating and Using a Mask
```python
num = np.arange(10)

# Create mask
mask = num > 5
print("Mask:")
print(mask)
# Output: [False False False False False False True True True True]

# Apply mask
result = num[mask]
print("Filtered result:")
print(result)
# Output: [6 7 8 9]
```

**Visual Representation:**
```
Array:  [0  1  2  3  4  5  6  7  8  9]
Mask:   [F  F  F  F  F  F  T  T  T  T]
Result: [               6  7  8  9]
```

##### Example 2: Reusing Masks
```python
arr1 = np.array([10, 20, 30, 40, 50])
arr2 = np.array([15, 25, 35, 45, 55])

# Create mask from arr1
mask = arr1 > 25
print("Mask:", mask)
# Output: [False False True True True]

# Apply to arr1
print("arr1 filtered:", arr1[mask])
# Output: [30 40 50]

# Apply same mask to arr2
print("arr2 filtered:", arr2[mask])
# Output: [35 45 55]
```

##### Example 3: Complex Masks
```python
num = np.arange(10)

# Multiple conditions
mask = (num > 2) & (num < 8) & (num % 2 == 0)
print("Mask:", mask)
# Output: [False False False False True False True False False False]

print("Result:", num[mask])
# Output: [4 6]
```

##### Example 4: Inverting Masks
```python
num = np.arange(10)

mask = num > 5
print("Original mask:", mask)
# Output: [False False False False False False True True True True]

inverted_mask = ~mask  # ~ inverts the mask
print("Inverted mask:", inverted_mask)
# Output: [True True True True True True False False False False]

print("Result:", num[inverted_mask])
# Output: [0 1 2 3 4 5]
```

---

### 🎪 Fancy Indexing

Fancy indexing allows you to select elements using arrays of indices.

#### Example 1: Basic Fancy Indexing
```python
num = np.arange(10)
indices = [0, 1, 4, 3, 4]
result = num[indices]
print(result)
# Output: [0 1 4 3 4]
```

**Visual Representation:**
```
Array:   [0  1  2  3  4  5  6  7  8  9]
Indices:  0  1     3  4     (pick these)
Result:  [0  1     3  4  4]
                     ↑  ↑ (can repeat!)
```

#### Example 2: Using NumPy Array as Index
```python
arr = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 2, 4])
result = arr[idx]
print(result)
# Output: [10 30 50]
```

#### Example 3: 2D Fancy Indexing
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select elements at (0,0), (1,1), (2,2)
rows = [0, 1, 2]
cols = [0, 1, 2]
result = arr_2d[rows, cols]
print(result)
# Output: [1 5 9]  (diagonal elements)
```

#### Example 4: Fancy Indexing with Boolean
```python
arr = np.array([10, 20, 30, 40, 50])

# Create boolean condition
condition = arr > 25

# Get indices where condition is True
indices = np.where(condition)[0]
print("Indices:", indices)
# Output: Indices: [2 3 4]

# Use fancy indexing
result = arr[indices]
print("Result:", result)
# Output: [30 40 50]
```

---

### 🔎 `np.where()` - Finding Indices

#### Example 1: Basic Usage
```python
num = np.arange(10)
where = np.where(num > 5)
print(where)
# Output: (array([6, 7, 8, 9]),)

# Access the actual array
indices = where[0]
print("Indices:", indices)
# Output: Indices: [6 7 8 9]
```
**Explanation:** Returns a **tuple** containing array of indices where condition is True

#### Example 2: Multiple Conditions
```python
num = np.arange(10)
indices = np.where((num > 2) & (num < 7))
print("Indices:", indices[0])
# Output: Indices: [3 4 5 6]
```

#### Example 3: 2D Array
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rows, cols = np.where(arr_2d > 5)
print("Row indices:", rows)
# Output: Row indices: [1 2 2 2]
print("Col indices:", cols)
# Output: Col indices: [2 0 1 2]

# These are positions: (1,2), (2,0), (2,1), (2,2)
# Values: 6, 7, 8, 9
```

---

### 🔀 Conditional Array with `np.where()`

The three-argument version of `np.where()` creates conditional transformations.

**Syntax:** `np.where(condition, value_if_true, value_if_false)`

#### Example 1: Conditional Transformation
```python
num = np.arange(10)
cond_arr = np.where(num > 5, num * 2, num)
print(cond_arr)
# Output: [0 1 2 3 4 5 12 14 16 18]
```

**Step-by-Step:**
```
Array:     [0  1  2  3  4  5  6   7   8   9]
Condition: [F  F  F  F  F  F  T   T   T   T]  (num > 5)
If True:                      12  14  16  18  (num * 2)
If False:  [0  1  2  3  4  5]                 (num)
Result:    [0  1  2  3  4  5  12  14  16  18]
```

#### Example 2: Replace with Constant
```python
num = np.arange(10)
result = np.where(num > 5, 100, num)
print(result)
# Output: [0 1 2 3 4 5 100 100 100 100]
```
**Explanation:** All values > 5 become 100, others stay same

#### Example 3: Two Different Transformations
```python
num = np.arange(10)
result = np.where(num % 2 == 0, num * 10, num * 100)
print(result)
# Output: [  0 100  20 300  40 500  60 700  80 900]
```
**Explanation:** 
- Even numbers multiply by 10
- Odd numbers multiply by 100

#### Example 4: No Transformation
```python
num = np.arange(10)
cond_arr = np.where(num > 5, num, num)
print(cond_arr)
# Output: [0 1 2 3 4 5 6 7 8 9]
```
**Explanation:** Both True and False return `num`, so no change

#### Example 5: String Labels
```python
num = np.arange(10)
labels = np.where(num > 5, "high", "low")
print(labels)
# Output: ['low' 'low' 'low' 'low' 'low' 'low' 'high' 'high' 'high' 'high']
```

**Visual:**
```
Array:     [0    1    2    3    4    5    6      7      8      9]
Condition: [F    F    F    F    F    F    T      T      T      T]
Result:    [low  low  low  low  low  low  high   high   high   high]
```

#### Example 6: Nested np.where()
```python
num = np.arange(10)
result = np.where(num < 3, "small",
                  np.where(num < 7, "medium", "large"))
print(result)
# Output: ['small' 'small' 'small' 'medium' 'medium' 'medium' 'medium'
#          'large' 'large' 'large']
```
**Explanation:** 
- If < 3: "small"
- Else if < 7: "medium"
- Else: "large"

#### Example 7: With 2D Arrays
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

result = np.where(arr_2d > 5, arr_2d, 0)
print(result)
# Output: [[0 0 0]
#          [0 0 6]
#          [7 8 9]]
```
**Explanation:** Replace values ≤ 5 with 0, keep others

---

### 📊 Filtering Methods Comparison

#### Complete Example
```python
num = np.arange(10)

# Method 1: Direct boolean filtering
method1 = num[num > 5]

# Method 2: Using mask
mask = num > 5
method2 = num[mask]

# Method 3: Using np.where (indices)
indices = np.where(num > 5)[0]
method3 = num[indices]

# Method 4: Using np.where (conditional)
method4 = np.where(num > 5, num, -1)

print("Method 1 (direct):", method1)
# Output: [6 7 8 9]
print("Method 2 (mask):", method2)
# Output: [6 7 8 9]
print("Method 3 (where indices):", method3)
# Output: [6 7 8 9]
print("Method 4 (where conditional):", method4)
# Output: [-1 -1 -1 -1 -1 -1  6  7  8  9]
```

---

### 💡 Filtering Quick Reference

| Operation | Syntax | Returns | Use Case |
|-----------|--------|---------|----------|
| **Boolean Filtering** | `arr[arr > 5]` | Filtered values | Quick filtering |
| **Mask** | `mask = arr > 5`<br>`arr[mask]` | Boolean array → Values | Reusable conditions |
| **Fancy Indexing** | `arr[[0,2,4]]` | Specific values | Select by position |
| **np.where (2 args)** | `np.where(arr > 5)` | Indices tuple | Find positions |
| **np.where (3 args)** | `np.where(arr > 5, x, y)` | Transformed values | Conditional transform |

---

## ➕➖ Adding and Removing Data

### Creating Arrays

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([4, 5, 6, 7])
arr3 = np.array([10, 11, 12, 13])
```

---

### 🔢 Combining Arrays

#### Element-wise Addition

##### Example 1: Adding Two Arrays
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([4, 5, 6, 7])

combined = arr1 + arr2
print(combined)
# Output: [5 7 9 11]
```

**Visual:**
```
arr1:     [1  2  3  4]
arr2:     [4  5  6  7]
         +------------
Result:   [5  7  9 11]
```

##### Example 2: Adding Three Arrays
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([4, 5, 6, 7])

combined = arr1 + arr2 + arr2
print(combined)
# Output: [9 12 15 18]
```

**Step-by-step:**
```
arr1:     [1  2  3  4]
arr2:     [4  5  6  7]
arr2:     [4  5  6  7]
         --------------
Result:   [9 12 15 18]
```

##### Example 3: Other Operations
```python
arr1 = np.array([10, 20, 30])
arr2 = np.array([1, 2, 3])

print("Subtraction:", arr1 - arr2)
# Output: Subtraction: [ 9 18 27]

print("Multiplication:", arr1 * arr2)
# Output: Multiplication: [10 40 90]

print("Division:", arr1 / arr2)
# Output: Division: [10. 10. 10.]
```

---

#### Concatenation (Joining Arrays)

##### Example 1: Basic Concatenation
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([4, 5, 6, 7])
arr3 = np.array([10, 11, 12, 13])

combined = np.concatenate((arr1, arr2, arr3))
print(combined)
# Output: [1 2 3 4 4 5 6 7 10 11 12 13]
```

**Visual:**
```
arr1: [1 2 3 4]
arr2:          [4 5 6 7]
arr3:                   [10 11 12 13]
Result: [1 2 3 4 4 5 6 7 10 11 12 13]
```

##### Example 2: Concatenating 2D Arrays (axis=0)
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

combined = np.concatenate((arr1, arr2), axis=0)
print(combined)
# Output: [[1 2]
#          [3 4]
#          [5 6]
#          [7 8]]
```
**Explanation:** Concatenate vertically (stack rows)

##### Example 3: Concatenating 2D Arrays (axis=1)
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

combined = np.concatenate((arr1, arr2), axis=1)
print(combined)
# Output: [[1 2 5 6]
#          [3 4 7 8]]
```
**Explanation:** Concatenate horizontally (stack columns)

---

### 🔍 Checking Shape Compatibility

##### Example 1: Compatible Shapes
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Shape a:", a.shape)  # (3,)
print("Shape b:", b.shape)  # (3,)
print("Compatible:", a.shape == b.shape)  # True

# Can perform element-wise operations
result = a + b
print("Result:", result)  # [5 7 9]
```

##### Example 2: Incompatible Shapes
```python
a = np.array([1, 2, 3])
c = np.array([7, 9])

print("Shape a:", a.shape)  # (3,)
print("Shape c:", c.shape)  # (2,)
print("Compatible:", a.shape == c.shape)  # False

# This would raise an error:
# result = a + c  # ValueError: operands could not be broadcast
```

##### Example 3: Multiple Shape Checks
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 9])

print(a.shape == b.shape == c.shape)
# Output: False

print(a.shape == b.shape)
# Output: True
```

---

### 📊 Adding Rows and Columns to 2D Arrays

#### Original Array

```python
original = np.array([[1, 2],
                     [3, 4]])
print("Original:")
print(original)
# Output: [[1 2]
#          [3 4]]
```

---

#### 🔥 `np.vstack()` - Adding Rows (Vertical Stack)

##### Example 1: Add One Row
```python
original = np.array([[1, 2],
                     [3, 4]])
newRow = np.array([[5, 6]])

with_new_row = np.vstack((original, newRow))
print(with_new_row)
# Output: [[1 2]
#          [3 4]
#          [5 6]]
```

**Visual:**
```
Original:  [[1 2]     New Row:  [[5 6]]
            [3 4]]

Result:    [[1 2]
            [3 4]
            [5 6]]  ← New row added at bottom
```

##### Example 2: Add Multiple Rows
```python
original = np.array([[1, 2],
                     [3, 4]])
row1 = np.array([[5, 6]])
row2 = np.array([[7, 8]])

result = np.vstack((original, row1, row2))
print(result)
# Output: [[1 2]
#          [3 4]
#          [5 6]
#          [7 8]]
```

##### Example 3: Add Row at Top
```python
original = np.array([[1, 2],
                     [3, 4]])
newRow = np.array([[0, 0]])

result = np.vstack((newRow, original))
print(result)
# Output: [[0 0]
#          [1 2]
#          [3 4]]
```

##### Example 4: vstack with Different Arrays
```python
arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[4, 5, 6]])
arr3 = np.array([[7, 8, 9]])

result = np.vstack((arr1, arr2, arr3))
print(result)
# Output: [[1 2 3]
#          [4 5 6]
#          [7 8 9]]
```

---

#### 🔥 `np.hstack()` - Adding Columns (Horizontal Stack)

##### Example 1: Add One Column
```python
original = np.array([[1, 2],
                     [3, 4]])
newCol = np.array([[7],
                   [8]])

with_new_col = np.hstack((original, newCol))
print(with_new_col)
# Output: [[1 2 7]
#          [3 4 8]]
```

**Visual:**
```
Original:  [[1 2]     New Col:  [[7]
            [3 4]]                [8]]

Result:    [[1 2 7]  ← New column added on right
            [3 4 8]]
```

##### Example 2: Add Multiple Columns
```python
original = np.array([[1, 2],
                     [3, 4]])
col1 = np.array([[5],
                 [6]])
col2 = np.array([[7],
                 [8]])

result = np.hstack((original, col1, col2))
print(result)
# Output: [[1 2 5 7]
#          [3 4 6 8]]
```

##### Example 3: Add Column at Beginning
```python
original = np.array([[1, 2],
                     [3, 4]])
newCol = np.array([[0],
                   [0]])

result = np.hstack((newCol, original))
print(result)
# Output: [[0 1 2]
#          [0 3 4]]
```

##### Example 4: Combine Multiple Arrays
```python
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([[4], [5], [6]])
arr3 = np.array([[7], [8], [9]])

result = np.hstack((arr1, arr2, arr3))
print(result)
# Output: [[1 4 7]
#          [2 5 8]
#          [3 6 9]]
```

---

### 💡 Stack Methods Quick Reference

```python
# Visual Memory Aid
vstack → Vertical   → Rows    ↓
hstack → Horizontal → Columns →
```

**Complete Example:**
```python
base = np.array([[5]])

# Add rows
with_rows = np.vstack((base, [[6]], [[7]]))
print("After vstack:")
print(with_rows)
# Output: [[5]
#          [6]
#          [7]]

# Add columns
with_cols = np.hstack((base, [[6]], [[7]]))
print("\nAfter hstack:")
print(with_cols)
# Output: [[5 6 7]]
```

---

### 🗑️ Deleting Elements

#### 1D Array Deletion

##### Example 1: Delete Single Element
```python
arr = np.array([1, 2, 3, 4, 5])
deleted = np.delete(arr, 2)
print(deleted)
# Output: [1 2 4 5]
```

**Visual:**
```
Original: [1  2  3  4  5]
           0  1  2  3  4  (indices)
                ↑
          Delete index 2

Result:   [1  2  4  5]
```

##### Example 2: Delete Multiple Elements
```python
arr = np.array([1, 2, 3, 4, 5, 6])
deleted = np.delete(arr, [0, 2, 4])
print(deleted)
# Output: [2 4 6]
```
**Explanation:** Deletes elements at indices 0, 2, and 4

##### Example 3: Delete Using Slice
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
deleted = np.delete(arr, slice(2, 6))
print(deleted)
# Output: [1 2 7 8]
```
**Explanation:** Deletes indices 2 through 5

##### Example 4: Delete from End
```python
arr = np.array([1, 2, 3, 4, 5])
deleted = np.delete(arr, -1)
print(deleted)
# Output: [1 2 3 4]
```

---

#### 2D Array Deletion

##### Example 1: Delete Row
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Delete row at index 1
deleted = np.delete(arr_2d, 1, axis=0)
print(deleted)
# Output: [[1 2 3]
#          [7 8 9]]
```

##### Example 2: Delete Column
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Delete column at index 1
deleted = np.delete(arr_2d, 1, axis=1)
print(deleted)
# Output: [[1 3]
#          [4 6]
#          [7 9]]
```

##### Example 3: Delete Multiple Rows
```python
arr_2d = np.array([[1, 2],
                   [3, 4],
                   [5, 6],
                   [7, 8]])

deleted = np.delete(arr_2d, [0, 2], axis=0)
print(deleted)
# Output: [[3 4]
#          [7 8]]
```

##### Example 4: Delete Multiple Columns
```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]])

deleted = np.delete(arr_2d, [1, 3], axis=1)
print(deleted)
# Output: [[1 3]
#          [5 7]]
```

---

### 📋 Operations Quick Reference

| Operation | Function | What it does | Example |
|-----------|----------|--------------|---------|
| **Element-wise add** | `arr1 + arr2` | Adds corresponding elements | `[1,2] + [3,4] = [4,6]` |
| **Concatenate 1D** | `np.concatenate((a,b))` | Joins arrays end-to-end | `[1,2] + [3,4] = [1,2,3,4]` |
| **Concatenate 2D (rows)** | `np.concatenate((a,b), axis=0)` | Stacks rows | Adds rows below |
| **Concatenate 2D (cols)** | `np.concatenate((a,b), axis=1)` | Stacks columns | Adds columns right |
| **Add rows** | `np.vstack((a,b))` | Stacks arrays vertically (↓) | Adds rows |
| **Add columns** | `np.hstack((a,b))` | Stacks arrays horizontally (→) | Adds columns |
| **Delete 1D** | `np.delete(arr, idx)` | Removes element at index | Removes specified indices |
| **Delete row** | `np.delete(arr, idx, axis=0)` | Removes row | Removes specified rows |
| **Delete column** | `np.delete(arr, idx, axis=1)` | Removes column | Removes specified columns |

---

### 🎯 Key Points Summary

1. **Element-wise operations** (`arr1 + arr2`): Perform math on corresponding elements
2. **Concatenate**: Joins arrays together (specify axis for 2D)
3. **vstack**: Adds rows vertically (↓)
4. **hstack**: Adds columns horizontally (→)
5. **delete**: Returns new array without specified elements (doesn't modify original)
6. **Shape compatibility**: Arrays must have compatible shapes for element-wise operations
7. **Axis parameter**:
   - `axis=0`: Operate on rows (vertical)
   - `axis=1`: Operate on columns (horizontal)

---

