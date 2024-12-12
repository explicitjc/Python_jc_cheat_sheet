# Python Help first part
In this repo you will find help for your python learnings. Feel free to use and share it. Don't hesitate to leave me a comment if you find it usefull.
Happy learning. 

# Python Data Prep and Cleaning

## 1. pandas
- **Use `pd.read_csv()` or `pd.read_excel()`** to import data from CSV or Excel files, respectively.
    - Example: 
    ```python
    df = pd.read_csv('data.csv')
    ```

## 2. SQL
- **Use `pd.read_sql()`** to import data from a SQL database.
    - Example:
    ```python
    df = pd.read_sql('SELECT * FROM table_name', conn)
    ```

## 3. JSON
- **Use `pd.read_json()`** to import data from a JSON file.
    - Example:
    ```python
    df = pd.read_json('data.json')
    ```

---

# Data Joining

## 1. Inner Join
- **Use `pd.merge()` with the `how='inner'` parameter** to perform an inner join.
    - Example:
    ```python
    df_result = pd.merge(df1, df2, on='common_column')
    ```

## 2. Left Join
- **Use `pd.merge()` with the `how='left'` parameter** to perform a left join.
    - Example:
    ```python
    df_result = pd.merge(df1, df2, on='common_column', how='left')
    ```

## 3. Right Join
- **Use `pd.merge()` with the `how='right'` parameter** to perform a right join.
    - Example:
    ```python
    df_result = pd.merge(df1, df2, on='common_column', how='right')
    ```

---

# Data Aggregation

## 1. GroupBy
- **Use `df.groupby()`** to group data by one or more columns and apply aggregation functions.
    - Example:
    ```python
    df_grouped = df.groupby('category')['value'].sum()
    ```

## 2. Pivot Tables
- **Use `pd.pivot_table()`** to create a pivot table from aggregated data.
    - Example:
    ```python
    pivot_table = pd.pivot_table(df_grouped, values='value', index='category', aggfunc='mean')
    ```

---

# Data Validation

## 1. Data Types
- **Use `pd.DataFrame.dtypes`** to check the data types of each column.
    - Example:
    ```python
    print(df.dtypes)
    ```

## 2. Missing Values
- **Use `pd.isna()` or `pd.notna()`** to identify missing values.
    - Example:
    ```python
    missing_values = df.isna().sum()
    ```

## 3. Outliers
- **Use statistical methods (e.g., Z-score, IQR)** to detect outliers.
    - Example:
    ```python
    outliers = df[(df['column'] > df['column'].quantile(0.99)) | (df['column'] < df['column'].quantile(0.01))]
    ```

---

# Data Cleaning

## 1. Handling Missing Values:
- **Use `pd.DataFrame.fillna()`** to fill missing values with a specific value (e.g., mean, median).
    - Example:
    ```python
    df.fillna(df.mean(), inplace=True)
    ```

- **Use `pd.DataFrame.dropna()`** to drop rows or columns with missing values.
    - Example:
    ```python
    df.dropna(inplace=True)
    ```

## 2. Removing Duplicates:
- **Use `pd.DataFrame.drop_duplicates()`** to remove duplicate rows.
    - Example:
    ```python
    df.drop_duplicates(inplace=True)
    ```

## 3. Data Normalization:
- **Use `pd.DataFrame.apply()`** with a normalization function (e.g., Min-Max Scaler) to scale data.
    - Example:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    ```

---

# Python Metrics Calculations

Here are the Python functions to calculate various metrics along with short examples:

## 1. Median
- **`statistics.median()`** (built-in): calculates the median of a list of numbers.
    - Example:
    ```python
    import statistics
    data = [1, 3, 5, 2, 4]
    median_value = statistics.median(data)
    print(median_value)  # Output: 3
    ```

## 2. Correlation (Pearson)
- **`numpy.corrcoef()`** (NumPy): calculates the Pearson correlation coefficient between two arrays.
    - Example:
    ```python
    import numpy as np
    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]
    corr_coef = np.corrcoef(x, y)[0, 1]
    print(corr_coef)  # Output: 0.923879532511287
    ```

## 3. Mean
- **`statistics.mean()`** (built-in): calculates the arithmetic mean of a list of numbers.
    - Example:
    ```python
    import statistics
    data = [1, 2, 3, 4]
    mean_value = statistics.mean(data)
    print(mean_value)  # Output: 2.5
    ```

## 4. Standard Deviation
- **`numpy.std()`** (NumPy): calculates the sample standard deviation of a list of numbers.
    - Example:
    ```python
    import numpy as np
    data = [1, 2, 3, 4]
    std_dev = np.std(data)
    print(std_dev)  # Output: 0.8164965809277349
    ```

## 5. Pearson Correlation Coefficient
- **`scipy.stats.pearsonr()`** (SciPy): calculates the Pearson correlation coefficient and p-value for two arrays.
    - Example:
    ```python
    from scipy.stats import pearsonr
    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]
    corr_coef, p_value = pearsonr(x, y)
    print(corr_coef)  # Output: 0.923879532511287
    ```

## 6. Spearman Rank Correlation Coefficient
- **`scipy.stats.spearmanr()`** (SciPy): calculates the Spearman rank correlation coefficient for two arrays.
    - Example:
    ```python
    from scipy.stats import spearmanr
    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]
    corr_coef = spearmanr(x, y)[0]
    print(corr_coef)  # Output: 0.923879532511287
    ```

---

# Classification Metrics

## 1. Precision
- **`precision_score(y_true, y_pred, average=None)`**
    - Example:
    ```python
    from sklearn.metrics import precision_score
    precision = precision_score(y_test, y_pred)
    ```

## 2. Recall
- **`recall_score(y_true, y_pred, average=None)`**
    - Example:
    ```python
    from sklearn.metrics import recall_score
    recall = recall_score(y_test, y_pred)
    ```

## 3. F1 Score
- **`f1_score(y_true, y_pred, average=None)`**
    - Example:
    ```python
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    ```

---

# Regression Metrics

## 1. Mean Squared Error (MSE)
- **`mean_squared_error(y_true, y_pred)`**
    - Example:
    ```python
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    ```

## 2. Mean Absolute Error (MAE)
- **`mean_absolute_error(y_true, y_pred)`**
    - Example:
    ```python
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    ```

## 3. Coefficient of Determination (R-squared)
- **`r2_score(y_true, y_pred)`**
    - Example:
    ```python
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    ```

---

# Python Data Visualization Functions

## 1. Histograms
- **`sns.histplot()`** from Seaborn: Creates a histogram to demonstrate the distribution of a single variable.
    - Example:
    ```python
    sns.histplot(x='total_bill', data=tips_data, kde=True, hue='sex')
    ```

## 2. Scatter Plots
- **`plt.scatter()`** from Matplotlib: Creates a scatter plot to demonstrate relationships between two variables.
    - Example:
    ```python
    plt.scatter(x, y, s=bubble_size, alpha=0.5, data=df)
    ```

## 3. Box Plots
- **`sns.boxplot()`** from Seaborn: Creates a box plot to demonstrate the shape of the distribution, central value, and variability of a single variable.
    - Example:
    ```python
    sns.boxplot(x='column_name', data=df)
    ```

## 4. Heatmaps
- **`sns.heatmap()`** from Seaborn: Creates a heatmap to demonstrate correlations between multiple variables.
    - Example:
    ```python
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    ```

## 5. WordClouds
- **`wordcloud` library**: Creates a word cloud to demonstrate the frequency or importance of words in text data.
    - Example: 
    ```python
    from wordcloud import WordCloud
    wordcloud = WordCloud().generate(text_data)
    ```

## 6. Interactive Charts
- **`bokeh` library**: Creates interactive charts, such as line plots, scatter plots, and histograms, to demonstrate the characteristics of data.
    - Example:
    ```python
    from bokeh.plotting import figure, show
    p = figure(title="Interactive Plot")
    p.line(x, y)
    show(p)
    ```

## 7. Bubble Charts
- **`plt.scatter()`** from Matplotlib: Creates a bubble chart to demonstrate relationships between three variables.
    - Example:
    ```python
    plt.scatter('X', 'Y', s='bubble_size', alpha=0.5, data=df)
    ```

## 8. Waterfall Charts
- **Not explicitly mentioned**, but can be created using libraries like `plotly` or `bokeh`.

## 9. KDE Plots
- **`sns.kdeplot()`** from Seaborn: Creates a kernel density estimate (KDE) plot to demonstrate the distribution of a single variable.
    - Example:
    ```python
    sns.kdeplot(x=iris["SepalLengthCm"], y=iris["PetalLengthCm"], linewidth=0.5, fill=True, multiple="layer", cbar=False, palette="crest", alpha=0.7)
    ```

---

# Python Functions for Data Analysis

Based on the provided search results, here’s a list of Python functions and examples to identify and reduce the impact of characteristics of data:

## 1. Handling Missing Values
- **`pandas.DataFrame.fillna()` and pandas.Series.fillna()** to replace missing values with a specified value (e.g., mean, median, or a custom value).
    - Example:
    ```python
    df['column_name'].fillna(df['column_name'].mean())
    ```

## 2. Data Normalization
- **`sklearn.preprocessing.MinMaxScaler`** to scale features to a common range (0-1).
    - Example:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    ```

## 3. Feature Selection
- **`pandas.DataFrame.drop()`** to remove columns with high correlation (>0.9) or low variance (<0.1).
    - Example:
    ```python
    high_corr_cols = [col for col in df.columns if df[col].corr(df['target_column']) > 0.9]
    df.drop(high_corr_cols, axis=1, inplace=True)
    ```

## 4. Handling Skewed Data
- **`scipy.stats.boxcox`** to transform skewed data using the Box-Cox transformation.
    - Example:
    ```python
    from scipy.stats import boxcox
    transformed_data = boxcox(data)
    ```

## 5. Identifying and Handling Outliers
- **`scipy.stats.zscore`** to calculate the Z-score for each data point.
    - Example:
    ```python
    from scipy.stats import zscore
    outliers = [i for i, zscore in enumerate(zscore(data)) if abs(zscore) > 3]
    data.drop(outliers, inplace=True)
    ```

## 6. Handling Categorical Data
- **`pandas.get_dummies`** to one-hot encode categorical variables.
    - Example:
    ```python
    one_hot_data = pd.get_dummies(data, columns=['categorical_column'])
    ```

## 7. Feature Engineering
- **`numpy.polyfit`** to create polynomial features from existing ones.
    - Example:
    ```python
    from numpy import polyfit
    poly_features = polyfit(data['x'], data['y'], 2)
    ```

## 8. Handling Multi-Collinearity
- **`sklearn.decomposition.PCA`** to reduce dimensionality using Principal Component Analysis (PCA).
    - Example:
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    ```

# EDA

# Unemployment Analysis and Data Cleaning

## 1. Data Overview
- **Display the first few rows, info, and summary statistics of the unemployment dataset.**
    - Example:
    ```python
    print(unemployment.head())
    print(unemployment.info())
    print(unemployment.describe())
    ```

- **Display the distribution of the 'continent' column.**
    - Example:
    ```python
    print(unemployment['continent'].value_counts())
    ```

---

## 2. Visualization

### Import required visualization libraries
- **Import the necessary libraries for visualization.**
    - Example:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```

### Create a histogram of 2021 unemployment rates, showing a full percent in each bin
- **Create a histogram to visualize unemployment rates for 2021.**
    - Example:
    ```python
    sns.histplot(data=unemployment, x="2021", binwidth=1)
    plt.show()
    ```

### Check data type issues for the 2019 column
- **Check the data type of the 2019 column.**
    - Example:
    ```python
    unemployment.info()  # Shows a wrong type for 2019
    ```

### Update the data type of the 2019 column to a float
- **Update the 2019 column to a float type.**
    - Example:
    ```python
    unemployment["2019"] = unemployment["2019"].astype(float)

    # Print the dtypes to check your work
    print(unemployment.dtypes)
    ```

---

## 3. Data Filtering

### Define a Boolean Series for countries outside Oceania
- **Create a filter to exclude Oceania countries from the dataset.**
    - Example:
    ```python
    not_oceania = ~unemployment["continent"].isin(["Oceania"])

    # Print unemployment without records related to countries in Oceania
    print(unemployment[not_oceania])
    ```

### Display the minimum and maximum unemployment rates during 2021
- **Display the minimum and maximum unemployment rates for the year 2021.**
    - Example:
    ```python
    print(unemployment["2021"].min(), unemployment["2021"].max())
    ```

---

## 4. Boxplot Visualization

### Create a boxplot of 2021 unemployment rates, broken down by continent
- **Create a boxplot to visualize the distribution of unemployment rates by continent.**
    - Example:
    ```python
    sns.boxplot(data=unemployment, x="2021", y="continent")
    plt.show()
    ```

---

# SUMMARIES WITH GROUPBY AND AGG

## 1. Print the mean and standard deviation of rates by year
- **Use aggregation to find the mean and standard deviation of unemployment rates.**
    - Example:
    ```python
    print(unemployment.agg(["mean", "std"]))
    ```

## 2. Print yearly mean and standard deviation grouped by continent
- **Group data by continent and calculate the mean and standard deviation.**
    - Example:
    ```python
    print(unemployment.groupby("continent").agg(["mean", "std"]))
    ```

## 3. Create a custom summary with grouped statistics by continent
- **Create a custom summary table with mean and standard deviation for 2021.**
    - Example:
    ```python
    continent_summary = unemployment.groupby("continent").agg(
        mean_rate_2021=("2021", "mean"),
        std_rate_2021=("2021", "std"))
    print(continent_summary)
    ```

### Create a bar plot of continents and their average unemployment
- **Visualize the average unemployment rate by continent using a bar plot.**
    - Example:
    ```python
    sns.barplot(data=unemployment, y="2021", x="continent")
    plt.show()
    ```

---

# ADDRESSING MISSING DATA

## 1. Count the number of missing values in each column
- **Check for missing values in the dataset.**
    - Example:
    ```python
    print(planes.isna().sum())
    ```

## 2. Find the five percent threshold
- **Calculate the threshold to drop columns with missing values below the threshold.**
    - Example:
    ```python
    threshold = len(planes) * 0.05
    ```

## 3. Create a filter and drop missing values
- **Drop rows or columns with missing values below the threshold.**
    - Example:
    ```python
    cols_to_drop = planes.columns[planes.isna().sum() <= threshold]

    # Drop missing values for columns below the threshold
    planes.dropna(subset=cols_to_drop, inplace=True)

    print(planes.isna().sum())
    ```

---

## 4. Handle missing "Price" values by using the median price per airline
- **Calculate and fill missing values in the "Price" column with the median price per airline.**
    - Example:
    ```python
    airline_prices = planes.groupby("Airline")["Price"].median()

    print(airline_prices)

    # Convert to a dictionary
    prices_dict = airline_prices.to_dict()

    # Map the dictionary to the missing values
    planes["Price"] = planes["Price"].fillna(planes["Airline"].map(prices_dict))

    # Check for missing values
    print(planes.isna().sum())
    ```

---

# Converting and Analyzing Categorical Data

## 1. Filter the DataFrame for object columns
- **Identify categorical columns in the dataset.**
    - Example:
    ```python
    non_numeric = planes.select_dtypes("object")
    ```

## 2. Loop through columns and print the number of unique values
- **Print the number of unique values for each categorical column.**
    - Example:
    ```python
    for col in non_numeric.columns:
        print(f"Number of unique values in {col} column: ", non_numeric[col].nunique())
    ```

---

# Flight Duration Categories

## 1. Define flight duration categories
- **Define regular flight duration categories for short, medium, and long flights.**
    - Example:
    ```python
    short_flights = "^0h|^1h|^2h|^3h|^4h"
    medium_flights = "^5h|^6h|^7h|^8h|^9h"
    long_flights = "10h|11h|12h|13h|14h|15h|16h"
    ```

## 2. Apply the conditions list to the flight categories
- **Classify flights based on duration into categories.**
    - Example:
    ```python
    conditions = [
        (planes["Duration"].str.contains(short_flights)),
        (planes["Duration"].str.contains(medium_flights)),
        (planes["Duration"].str.contains(long_flights))
    ]

    planes["Duration_Category"] = np.select(conditions, 
                                            flight_categories,
                                            default="Extreme duration")

    # Plot the counts of each category
    sns.countplot(data=planes, x="Duration_Category")
    plt.show()
    ```

---

# Handling Duration and Price

## 1. Clean and convert flight duration
- **Remove string characters and convert duration to a float.**
    - Example:
    ```python
    planes["Duration"] = planes["Duration"].astype(str).str.replace("h", "")
    planes["Duration"] = planes["Duration"].astype(float)
    ```

## 2. Plot a histogram of flight duration
- **Visualize the distribution of flight durations.**
    - Example:
    ```python
    sns.histplot(data=planes, x="Duration")
    plt.show()
    ```

---

# Price and Duration by Airline

## 1. Price standard deviation by airline
- **Calculate the standard deviation of flight prices by airline.**
    - Example:
    ```python
    planes["airline_price_st_dev"] = planes.groupby("Airline")["Price"].transform(lambda x: x.std())

    print(planes[["Airline", "airline_price_st_dev"]].value_counts())
    ```

## 2. Median duration by airline
- **Calculate the median flight duration by airline.**
    - Example:
    ```python
    planes["airline_median_duration"] = planes.groupby("Airline")["Duration"].transform(lambda x: x.median())

    print(planes[["Airline","airline_median_duration"]].value_counts())
    ```

## 3. Mean price by destination
- **Calculate the mean price of flights by destination.**
    - Example:
    ```python
    planes["price_destination_mean"] = planes.groupby("Destination")["Price"].transform(lambda x: x.mean())

    print(planes[["Destination","price_destination_mean"]].value_counts())
    ```

---

# Outliers

## 1. Plot a histogram of flight prices
- **Create a histogram to visualize the distribution of flight prices.**
    - Example:
    ```python
    sns.histplot(data=planes, x="Price")
    plt.show()
    ```

## 2. Display descriptive statistics for flight duration
- **Show the descriptive statistics for the flight duration column.**
    - Example:
    ```python
    print(planes["Duration"].describe())
    ```

## 3. Detect outliers in flight prices
- **Calculate and remove outliers from the flight price column using the IQR method.**
    - Example:
    ```python
    price_seventy_fifth = planes["Price"].quantile(0.75)
    price_twenty_fifth = planes["Price"].quantile(0.25)

    prices_iqr = price_seventy_fifth - price_twenty_fifth

    upper = price_seventy_fifth + (1.5 * prices_iqr)
    lower = price_twenty_fifth - (1.5 * prices_iqr)

    planes = planes[(planes["Price"] > lower) & (planes["Price"] < upper)]

    print(planes["Price"].describe())
    ```

## **Contact me for th second part** 
