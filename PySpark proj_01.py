# Install the PySpark library if not already installed
!pip install pyspark


# Initialize Spark using SparkContext
import pyspark as sp
sc = sp.SparkContext.getOrCreate()  
print(sc)  
print(sc.version)  


# Create a SparkSession, the main entry point for PySpark applications
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")  
print(spark)


# Import necessary libraries
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import round, mean, stddev, when, col, abs as pyspark_abs


# Load airport data from CSV file
file_path = '/content/airports.csv'
airports = spark.read.csv(/content/airports.csv, header=True ,inferSchema=True )  
airports.show()  

# Load flight data from CSV file
flights = spark.read.csv('/content/flights_small.csv', header=True, inferSchema=True)  

flights.show() 


# Load plane data from CSV file and rename the year column
planes = spark.read.csv('/content/planes.csv', header=True , inferSchema=True)  
planes = planes.withColumnRenamed('year', 'plane_year')  
planes.show() 


# Catalog checks, i.e., listing available databases and tables
spark.catalog.listDatabases()  
spark.catalog.listTables()  


# Create a temporary view for SQL operations so we can run SQL queries on it
flights.createOrReplaceTempView('flights')  
airports.createOrReplaceTempView('airports')  




flights.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in flights.columns]).show()


# Data transformation and column calculations
flights = flights.selectExpr("distance / 60 as duration_hrs", "CAST(distance AS float) as distance", 
                             "CAST(air_time AS float) as air_time", "CAST(dep_delay AS float) as dep_delay")

flights = flights.repartition("origin")


flights.cache()                             


# Filter operations for long flights, selecting flights with a distance greater than 1000 miles
long_flights = flights.filter(flights.distance > 1000) 
long_flights.show()  


# Selection and filtering of specific columns, filtering flights from SEA to PDX
selected_cols = flights.select('tailnum', 'origin', 'dest') 
filtered_flights = flights.filter(flights.origin == 'SEA').filter(flights.dest == 'PDX')  
filtered_flights.show()  

# Calculating average speed using the distance and air_time columns
avg_speed = round(flights.distance / (flights.air_time / 60), 2).alias("avg_speed")  
speed_1 = flights.select('origin', 'dest', 'tailnum', avg_speed)  
speed_1.show()  

# Performing statistical analysis on the air_time and distance columns
flights.describe(['air_time', 'distance']).show()  

# Aggregate queries to find minimum distance and average air_time
flights.filter(flights.origin == 'SEA').groupBy().min('distance').show()  
flights.filter(flights.origin == 'SEA').groupBy().avg('air_time').show()  

# Joining flight data with airport data
airports = airports.withColumnRenamed('faa', 'dest')  
flights_with_airports = flights.join(airports, on='dest', how='leftouter')  

# Joining flight data with plane data
model_data_with_outliers = flights_with_airports.join(planes, on='tailnum', how='leftouter')  
model_data_with_outliers.show()  

# Feature engineering: calculating plane age and determining if the flight is late
model_data_with_outliers = model_data_with_outliers.withColumn('plane_age', model_data_with_outliers.plane_year - model_data_with_outliers.year)  
model_data_with_outliers = model_data_with_outliers.withColumn('is_late', model_data_with_outliers.arr_delay > 0)  
model_data_with_outliers = model_data_with_outliers.withColumn('label', model_data_with_outliers.is_late.cast('integer'))  

# Filling missing values with the mean plane year
mean_plane_year = model_data_with_outliers.select(mean('plane_year')).first()[0]  
model_data_with_outliers = model_data_with_outliers.fillna({'plane_year': mean_plane_year}) 

# Function to detect outliers based on Z-score
def detect_outliers(df, column, threshold=3):
    mean_val, stddev_val = df.select(mean(column), stddev(column)).first()  
    z_score_col = (col(column) - mean_val) / stddev_val  
    return df.withColumn('is_outlier', when(pyspark_abs(z_score_col) > threshold, 1).otherwise(0))  

# Detecting outliers in the distance column
model_data_with_outliers = detect_outliers(model_data_with_outliers, 'distance')  

# Additional calculations such as interaction between distance and air_time
model_data_with_outliers = model_data_with_outliers.withColumn('distance_airtime_interaction', model_data_with_outliers['distance'] * model_data_with_outliers['air_time'])  
model_data_with_outliers = model_data_with_outliers.withColumn('arr_delay_winsorized',
                                   when(col('arr_delay') > 100, 100)  
                                   .when(col('arr_delay') < -100, -100)
                                   .otherwise(col('arr_delay')))  

# SQL queries to gain insights into the data
spark.sql("SELECT dest, COUNT(*) AS num_flights FROM flights GROUP BY dest ORDER BY num_flights DESC").show()  
spark.sql("SELECT carrier, AVG(arr_delay) AS avg_delay FROM flights WHERE distance > 1000 GROUP BY carrier ORDER BY avg_delay DESC").show()  
spark.sql("SELECT dest, AVG(air_time) AS avg_air_time FROM flights GROUP BY dest ORDER BY avg_air_time DESC LIMIT 5").show()  
spark.sql("SELECT carrier, (SUM(CASE WHEN arr_delay >= 15 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS delay_percentage FROM flights GROUP BY carrier ORDER BY delay_percentage DESC").show()  
spark.sql("SELECT carrier, AVG(plane_age) AS avg_plane_age FROM flights WHERE dest = 'JFK' GROUP BY carrier ORDER BY avg_plane_age DESC").show()  

# Additional Queries 
 missing_values = spark.sql("SELECT COUNT(CASE WHEN dep_delay IS NULL THEN 1 END) AS missing_dep_delay, COUNT(CASE WHEN arr_delay IS NULL THEN 1 END) AS missing_arr_delay FROM flights"); missing_values.show()

 avg_distance_speed = spark.sql("SELECT origin, dest, ROUND(AVG(distance), 2) AS avg_distance, ROUND(AVG(distance / (air_time / 60)), 2) AS avg_speed FROM flights GROUP BY origin, dest"); avg_distance_speed.show()

 sea_flights = spark.sql("SELECT tailnum, origin, dest FROM flights WHERE origin = 'SEA' AND dest = 'PDX'"); sea_flights.show()

 flight_stats = spark.sql("SELECT MIN(distance) AS min_distance, MAX(distance) AS max_distance, AVG(air_time) AS avg_air_time FROM flights WHERE origin = 'SEA'"); flight_stats.show()

 joined_data = spark.sql("SELECT f.origin, f.dest, f.distance, f.air_time, f.tailnum, a.name AS dest_airport_name, p.manufacturer, p.plane_year FROM flights f LEFT JOIN airports a ON f.dest = a.faa LEFT JOIN planes p ON f.tailnum = p.tailnum"); joined_data.show()

 joined_data.createOrReplaceTempView("model_data_with_outliers")

 plane_ages = spark.sql("SELECT tailnum, plane_year, 2023 - plane_year AS plane_age FROM planes WHERE plane_year IS NOT NULL"); plane_ages.show()

 mean_plane_year = spark.sql("SELECT AVG(plane_year) as avg_year FROM planes WHERE plane_year IS NOT NULL").first()[0]; model_data_with_outliers = model_data_with_outliers.na.fill({'plane_year': mean_plane_year})

 outliers = spark.sql(f"SELECT *, CASE WHEN ABS((distance - {mean_plane_year}) / {stddev_val}) > 3 THEN 1 ELSE 0 END AS is_outlier FROM model_data_with_outliers"); outliers.show()
 
 distance_analysis = spark.sql("SELECT origin, dest, COUNT(*) AS total_flights, AVG(distance) AS avg_distance FROM flights GROUP BY origin, dest"); distance_analysis.show()


# Pipeline for categorical encoding and feature assembly
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Setting up service carrier encoding
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index') 
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carr_fact')  

# Setting up destination encoding
dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')  
dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact') 

# Assembling features into a single column
vec_assembler = VectorAssembler(inputCols=['month', 'air_time', 'carr_fact', 'dest_fact', 'plane_age'],
                                outputCol='features', handleInvalid="skip")  

# Creating the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])  
piped_data = flights_pipe.fit(model_data_with_outliers).transform(model_data_with_outliers)  
piped_data.show()  

# Splitting the data into training and testing sets
training, test = piped_data.randomSplit([.6, .4])  

# Model training and cross-validation process
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune

# Setting up logistic regression model
lr = LogisticRegression()  
evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')  

# Setting up parameter grid to try out a range of parameters
grid = tune.ParamGridBuilder() 
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))  
grid = grid.addGrid(lr.elasticNetParam, [0, 1])  
grid = grid.build()  

# Creating cross-validation using the model and parameter grid
cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)  
models = cv.fit(training)  

# Evaluating the model
best_lr = models.bestModel  
test_results = best_lr.transform(test)  
print(evaluator.evaluate(test_results))  

# Save the model
# best_lr.save("path/to/save/your/model")  

# Exporting the final table to SQL Server
# model_data_with_outliers.write.format("jdbc").option("url", "jdbc:sqlserver://your_server_name;databaseName=your_db_name") \
# .option("dbtable", "final_table_name").option("user", "your_username").option("password", "your_password").mode("overwrite").save()
