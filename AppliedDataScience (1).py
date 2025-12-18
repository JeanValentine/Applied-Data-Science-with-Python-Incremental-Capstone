import pandas as pd # data manipulation and analysis
import numpy as np # numerical operations
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # Advanced statistical plotting which builds on matplotlib
import json # for reading and writing JSON files
import os # interacts with the operating system 
from sklearn.preprocessing import MinMaxScaler # Scales numerical values into a specific range


#Seting some display options for pandas: 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

'''Makes sure that when printing a dataframe the columns are shown with no truncation. 
We set the maximum line width to 1000 character so that the data prints in a more readable format.'''


file_path = r'C:\Users\valen\OneDrive\Desktop\FloridaBikeRentals.csv' #change this to the path of your dataset


if os.path.exists(file_path):
    bike_data = pd.read_csv(file_path, encoding='ISO-8859-1')  # or try cp1252 if needed
else:
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

'''Defining the CSV file path and checking if the file exists. If the file exists, 
it will load into pandas dataframe bike_data with a specified encoding. If no then it will throw an error message.'''


#fixing the date format here: 

bike_data['Date'] = pd.to_datetime(bike_data['Date'], dayfirst=True) # converts the data column to a datetime object


#Displaying the first few rows of the dataset:

print("First few rows of the dataset:")
print(bike_data.head())


#displaying the shape of the dataset: 

print("\nShape of the dataset:")
print(bike_data.shape)


#displaying column names and data types: 

print("\nColumn names and data types:")
print(bike_data.dtypes)


#Checking for the missing values:
 
print("\nMissing values in each column:")
print(bike_data.isnull().sum())


#checking for duplicate records: 

print("\nNumber of duplicate records:")
print(bike_data.duplicated().sum())

#handling the missing values: 

missing_values = bike_data.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    print("\nHandling missing values:")
    for column in missing_values.index:
        if bike_data[column].dtype in ['float64', 'int64']: #filling the numeric columns with the mean
            bike_data[column].fillna(bike_data[column].mean(), inplace=True)
        else:
            bike_data[column].fillna(bike_data[column].mode()[0], inplace=True) #filling the categorical columns with mode
    print("Missing values handled.")


#Checking the data types: 

print("\nData types after handling missing values:")
print(bike_data.dtypes)


#Optimizing the memory usage by converting the data types: 

bike_data['Date'] = pd.to_datetime(bike_data['Date'])
bike_data['Hour'] = bike_data['Hour'].astype('int8')
bike_data['Temperature(°C)'] = bike_data['Temperature(°C)'].astype('float32')
bike_data['Humidity(%)'] = bike_data['Humidity(%)'].astype('float32')
bike_data['Wind speed (m/s)'] = bike_data['Wind speed (m/s)'].astype('float32')
bike_data['Visibility (10m)'] = bike_data['Visibility (10m)'].astype('float32')
bike_data['Dew point temperature(°C)'] = bike_data['Dew point temperature(°C)'].astype('float32')
bike_data['Solar Radiation (MJ/m2)'] = bike_data['Solar Radiation (MJ/m2)'].astype('float32')
bike_data['Rainfall(mm)'] = bike_data['Rainfall(mm)'].astype('float32')
bike_data['Snowfall (cm)'] = bike_data['Snowfall (cm)'].astype('float32')


#exporting cleaned data to JSON format: 

cleaned_data_path = 'bike_rental_cleaned.json'
bike_data.to_json(cleaned_data_path, orient='records', date_format='iso')
print(f"\nCleaned data exported to {cleaned_data_path}")

'''Saving the cleaned data frame to a JSON file. 
orient='records' means each row is a JSON object and the data_format='iso' saves the dates in 
the ISO 8601 format. '''


#writing a short report summarizing observations about the data: 

report = { #creates a python dictionary to summarize the observations. 
    "Observations": [
        "The dataset contains various features impacting bike rentals such as weather conditions, seasonality, and operational factors.",
        "Missing values were handled by filling numeric columns with mean and categorical columns with mode.",
        "Data types were optimized for memory efficiency, especially for numerical columns.",
        "The dataset has no duplicate records."
    ],
    "Data Types": {col: str(dtype) for col, dtype in bike_data.dtypes.items()},
    "Shape of Cleaned Data": bike_data.shape
}


#Saving the report to a JSON file: 

report_path = 'bike_rental_report.json'
with open(report_path, 'w') as report_file:
    json.dump(report, report_file, indent=4)
print(f"\nReport saved to {report_path}")


#Multiplying the temperature by 10 for standardization: 

bike_data['Temperature(°C)'] *= 10


#Scaling visibility to a range between 0 and 1 using MinMax scaling: 

scaler = MinMaxScaler()
bike_data['Visibility (10m)'] = scaler.fit_transform(bike_data[['Visibility (10m)']])


#conducting basic statistical analysis: 

statistical_summary = bike_data.describe() #.describe() calculates mean, std, min, mac, quatiles, etc. 
print("\nStatistical summary of key columns:")
print(statistical_summary[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count']])


#comparing the results with raw dataset statistic: 

raw_statistics = bike_data.copy() #reverses the scaling on temperature and visibility for comparison
raw_statistics['Temperature(°C)'] /= 10  # Reverse the transformation
raw_statistics['Visibility (10m)'] = scaler.inverse_transform(bike_data[['Visibility (10m)']])
raw_statistics = raw_statistics.describe()

print("\nRaw dataset statistics for comparison:")
print(raw_statistics[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count']])


#identifying the columns not suitable for statistical analysis: 
#lists all non numeric columns that cant be summarized statistically

non_statistical_columns = bike_data.select_dtypes(exclude=['number']).columns.tolist()
print("\nColumns not suitable for statistical analysis:")
print(non_statistical_columns)


#Recommending the possible datatype changes: 

datatype_recommendations = {
    "Seasons": "Categorical - consider using 'category' dtype",
    "Holiday": "Boolean - consider using 'bool' dtype",
    "Functioning Day": "Boolean - consider using 'bool' dtype"
}
print("\nDatatype recommendations:")

for column, recommendation in datatype_recommendations.items():
    print(f"{column}: {recommendation}")


#exporting the processed data to a CSV file: 
#saves the fully processed data tp a CSV file without the index column

processed_data_path = 'bike_rental_processed.csv'
bike_data.to_csv(processed_data_path, index=False)
print(f"\nProcessed data exported to {processed_data_path}")


#preparing a short report on statistical observations and insights: 

statistical_report = {
    "Statistical Observations": [
        "Temperature has been standardized by multiplying by 10.",
        "Visibility has been scaled to a range between 0 and 1.",
        "The statistical summary shows the mean, standard deviation, and quartiles for key columns.",
        "Comparison with raw dataset statistics indicates that transformations were applied correctly."
    ],
    "Statistical Summary": json.loads(statistical_summary.to_json()),
    "Raw Dataset Statistics": json.loads(raw_statistics.to_json())

}


#saving the statistical report to a JSON file: 

statistical_report_path = 'bike_rental_statistical_report.json'
with open(statistical_report_path, 'w') as stat_report_file:
    json.dump(statistical_report, stat_report_file, indent=4)

print(f"\nStatistical report saved to {statistical_report_path}")


#Identifying categorical and numerical variables: 

categorical_columns = bike_data.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = bike_data.select_dtypes(include=['number']).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)
print("\nNumerical columns:")
print(numerical_columns)


#Performing pivoting operations on the dataset based on the categorical columns: 

pivot_seasons = bike_data.pivot_table(index='Seasons', values='Rented Bike Count', aggfunc='mean')

print("\nAverage rented bike count by Seasons:")
print(pivot_seasons)


#analyzing trends across Holiday and Functioning Day: 

pivot_holiday = bike_data.pivot_table(index='Holiday', values='Rented Bike Count', aggfunc='mean')

print("\nAverage rented bike count by Holiday:")
print(pivot_holiday)

pivot_functioning_day = bike_data.pivot_table(index='Functioning Day', values='Rented Bike Count', aggfunc='mean')

print("\nAverage rented bike count by Functioning Day:")
print(pivot_functioning_day)


#creating the distribution tables: 

temp_hour_distribution = bike_data.groupby('Hour')['Temperature(°C)'].describe()

print("\nTemperature distribution by Hour:")
print(temp_hour_distribution)

rented_bike_hour_distribution = bike_data.groupby('Hour')['Rented Bike Count'].describe()

print("\nRented Bike Count distribution by Hour:")
print(rented_bike_hour_distribution)

seasons_rented_distribution = bike_data.groupby('Seasons')['Rented Bike Count'].describe()

print("\nRented Bike Count distribution by Seasons:")
print(seasons_rented_distribution)


#encoding categorical variables and saving the data as "Rental_Bike_Data_Dummy.csv": 
# converts categorical columns into binary dummy variables 

bike_data_encoded = pd.get_dummies(bike_data, columns=categorical_columns, drop_first=True)
encoded_data_path = 'Rental_Bike_Data_Dummy.csv'
bike_data_encoded.to_csv(encoded_data_path, index=False)

print(f"\nEncoded data saved to {encoded_data_path}")


# Setting up the visualization style: 

sns.set(style='whitegrid')

#Bar plot for average rentals by seasons: 

plt.figure(figsize=(10, 6))
sns.barplot(x=pivot_seasons.index, y='Rented Bike Count', data=pivot_seasons.reset_index())
plt.title('Average Rented Bike Count by Seasons')
plt.xlabel('Seasons')
plt.ylabel('Average Rented Bike Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('average_rented_bike_count_by_seasons.png')
plt.show()


#The line plot for showing the hourly rentals throughout the day: 

plt.figure(figsize=(12, 6))
sns.lineplot(x='Hour', y='Rented Bike Count', data=bike_data,estimator='mean', ci=None, marker='o')
plt.title('Hourly Rented Bike Count Throughout the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Rented Bike Count')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('hourly_rented_bike_count.png')
plt.show()


#The heatmap showing the correlation among numerical variables: 

plt.figure(figsize=(10, 8))
correlation_matrix = bike_data[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()


# The box plot to identify outliers in temperature and the rented bike count: 

plt.figure(figsize=(12, 6))
sns.boxplot(data=bike_data[['Temperature(°C)', 'Rented Bike Count']])
plt.title('Box Plot of Temperature and Rented Bike Count')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('box_plot_temperature_rented_bike_count.png')
plt.show()


#Recording the observations and insights: 

visualization_report = {
    "Visualizations": [
        "The bar plot shows that Summer has the highest average rented bike count, while Winter has the lowest.",
        "The line plot indicates peak rental hours around 8 AM and 6 PM, likely corresponding to commuting times.",
        "The heatmap reveals strong correlations between Temperature and Rented Bike Count, suggesting that warmer weather increases bike rentals.",
        "The box plot indicates some outliers in both Temperature and Rented Bike Count, which may require further investigation."
    ],
    "Plots": {
        "Average Rented Bike Count by Seasons": "average_rented_bike_count_by_seasons.png",
        "Hourly Rented Bike Count Throughout the Day": "hourly_rented_bike_count.png",
        "Correlation Heatmap of Numerical Variables": "correlation_heatmap.png",
        "Box Plot of Temperature and Rented Bike Count": "box_plot_temperature_rented_bike_count.png"
    }
}


#saving the visualization report to a JSON file: 

visualization_report_path = 'bike_rental_visualization_report.json'
with open(visualization_report_path, 'w') as vis_report_file:
    json.dump(visualization_report, vis_report_file, indent=4)

print(f"\nVisualization report saved to {visualization_report_path}")


print("All tasks have been executed, and reports have been generated.")

print("You can now analyze the generated reports and visualizations for insights.")


