# Iris Classification Analysis

## Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
  - [Steps](#steps)
- [Data Overview](#data-overview)
- [Python Script](#python-script)
- [Results](#results)
- [Visualizations](#visualizations)
  - [Python Script](#python-script)
  - [Pairplot of Iris Dataset](#pair-plot-of-iris-dataset)
  - [Feature Importance ](#feature-importance)
- [Conclusion](#conclusion)
- [Assignment Details](#assignment-details)
- [Instructions for the Assignment](#instructions-for-the-assignment)
- [Contact Information](#contact-information)

## Introduction
This is a project for the STQD6324 Data Management course, in which we apply the Spark MLlib library to classify the Iris dataset. The purpose is to predict the Iris flower species based on sepal and petal length and width features using Decision Tree Classifier algorithm and measure its accuracy.

## Methodology
The steps followed in this project are:
1. **Load the Dataset**: Load the Iris dataset into a Spark DataFrame.
2. **Data Preprocessing**: Split the dataset into training and testing sets.
3. **Model Building**: Select and train a Decision Tree Classifier using a pipeline.
4. **Hyperparameter Tuning**: Use cross-validation and grid search to fine-tune the model's hyperparameters.
5. **Model Evaluation**: Evaluate the model's performance using accuracy, precision, recall, and F1 score metrics.
6. **Visualization**: Generate visualizations to understand the data and feature importance.

## How to Run

- Make sure to have Hadoop, Spark, and the necessary libraries installed and configured on your system.
- Have access to the Ambari dashboard for monitoring.
- Have R installed on your local machine.

### Steps
1. **Get the Iris Dataset from R**:
   - Open R or RStudio on the local machine.
   - Run the following R code to save the Iris dataset as a CSV file:
     ```r
     # Load the iris dataset
     data(iris)

     # Save the dataset to a CSV file
     write.csv(iris, "path/to/save/iris.csv", row.names = FALSE)
     ```

2. **Transfer the Dataset to the Hadoop Server**:
   - Open Command Prompt on the local machine.
   - Navigate to the directory containing the `iris.csv` file:
     ```bash
     cd "path/to/your/csv/file"
     ```
   - Use the `scp` command to transfer the dataset file to the Hadoop server (replace `hadoop_ip_address` with your actual IP address and `your_port` with your actual port number if not default):
     ```bash
     scp -P your_port iris.csv maria_dev@hadoop_ip_address:/home/maria_dev/Aishah/iris.csv
     ```

3. **Upload the File to HDFS**:
   - Log in to the Hadoop server using PuTTY:
     ```bash
     ssh -p your_port maria_dev@your_hadoop_ip_address
     ```
   - Upload the file to HDFS:
     ```bash
     hdfs dfs -put /home/maria_dev/Aishah/iris.csv /user/maria_dev/iris.csv
     ```

4. **Create and Run the PySpark Script**:
   - Ensure you have the `iris_classification.py` script on the Hadoop server. Transfer it using `scp` if necessary:
     ```bash
     scp -P your_port /path/to/your/script/iris_classification.py maria_dev@your_hadoop_ip_address:/home/maria_dev/Aishah/
     ```
   - Log in to the Hadoop server using PuTTY if not already logged in:
     ```bash
     ssh -p your_port maria_dev@your_hadoop_ip_address
     ```
   - Navigate to the directory containing the script:
     ```bash
     cd /home/maria_dev/Aishah
     ```
   - Run the script using Spark:
     ```bash
     spark-submit iris_classification.py
     ```

5. **Access Ambari Dashboard**:
   - Open Ambari in the web browser using the appropriate URL (`http://127.0.0.1:1080`).
   - Monitor the Spark and start the service actions using the Ambari dashboard.

## Data Overview
### Top 5 Rows of the Dataset
The following table shows the first five rows of the Iris dataset:

![Head Data](/Assignment_3/output/Top_5_head_data.png)

The data set is about iris flowers and it has four features namely sepal length, sepal width, petal length, petal width and one target variable which is the species of the flower. The first five rows all refer to the species Iris-setosa. These measurements will be used in the identification of the flowers to their various species using the machine learning algorithms.

## Python Script
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

# Define schema for Iris dataset
schema = StructType([
    StructField("petal_length", FloatType(), True),
    StructField("petal_width", FloatType(), True),
    StructField("species", StringType(), True)
])

# Load the dataset
iris_df = spark.read.csv("/user/maria_dev/Aishah/iris.csv", header=False, schema=schema)

# Show the head of the data
iris_df.show(5)

# Split the dataset into training (80%) and testing (20%) sets
train_df, test_df = iris_df.randomSplit([0.8, 0.2], seed=1234)

# Convert labels to indexed labels
indexer = StringIndexer(inputCol="species", outputCol="label")

# Assemble feature columns
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")

# Initialize Decision Tree classifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Create a Pipeline
pipeline = Pipeline(stages=[indexer, assembler, dt])

# Create parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [2, 4, 6, 8, 10]) \
    .addGrid(dt.maxBins, [20, 30, 40, 50]) \
    .build()

# Create evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Create CrossValidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Fit the model
cvModel = crossval.fit(train_df)

# Make predictions on test data
predictions = cvModel.transform(test_df)

# Evaluate accuracy
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy: {:.4f}".format(accuracy))

# Additional evaluation metrics
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

# Stop the Spark session
spark.stop()
```

## Results & Discussions
![Model Evaluation](/Assignment_3/output/results.png)

The model achieved the following metrics:
- Test Accuracy: 0.9583

The model correctly classified approximately 95.83% of the test samples. The high accuracy level shows that the model can differentiate between the three species of Iris flowers.

- Precision: 0.9635

Precision measures how many of the predicted positive samples are positive. A precision of 96.35% means that when the model predicts a certain species, it is correct about 96.35% of the time.

- Recall: 0.9583

Recall measures the percentage of actual positive samples that the model correctly classifies. A recall of 95. 83% means that the model correctly identifies 95.83% of the actual samples of a species.

- F1 Score: 0.9581

The F1 Score is the average of the precision and recall, and it offers a single value that considers both. The F1 Score was calculated to be 95.81%, which shows the high reliability and accuracy of the model in predicting the results.

## Visualizations
### Python Script 
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plotting pairplot of the dataset
pairplot = sns.pairplot(iris_df, hue='species')
pairplot.fig.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.savefig("pairplot.png")
plt.show()

# Plot feature importance
importances = cvModel.bestModel.stages[-1].featureImportances
indices = np.argsort(importances)[::-1]
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
colors = sns.color_palette("viridis", len(importances))

plt.figure(figsize=(10, 5))
plt.title("Feature Importances")

# Convert numpy.int64 to int for indexing
plt.bar(range(len(importances)), [importances[int(i)] for i in indices], color=colors, align="center")
plt.xticks(range(len(importances)), [features[i] for i in indices])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig("feature_importance.png")
plt.show()
```
### Pairplot of Iris Dataset
![Pairplot](/Assignment_3/output/pairplot.png)

Species Distribution:
   - Iris-setosa (blue): Distinct clusters, especially in petal length and width.
   - Iris-versicolor (orange) and Iris-virginica (green): There is more overlap, especially in sepal length and sepal width, but there is some separation in petal length and petal width.

Feature Relationships:
   - Petal length vs. Petal width: Clear separation between species, especially between setosa and the other two.
   - Sepal length vs. Sepal width: More overlap makes distinguishing between versicolor and virginica harder.

Density Plots: Petal length and petal width show the most distinct distributions.

### Feature Importance  
![Feature Importance](/Assignment_3/output/feature_importance.png)

Petal length: The most important feature with the highest importance score, playing a significant role in distinguishing species.

Petal width: Second most important feature.

Sepal length and Sepal width: Less important compared to petal measurements.

The plot helps us understand which features are most important in predicting the species of an iris flower. Petal measurements, such as length and width, are much more important than sepal measurements.

## Conclusion

In case of the Iris dataset, the Decision Tree Classifier was able to give good results. Potential future work could include experimenting with other classifiers and feature engineering methods to enhance the model performance.

## Assignment Details

- Course: STQD6324 Data Management
- Semester: 2 2023/2024
- Due Date: 2024-07-01

## Instructions for the Assignment

1. **Load the Iris** dataset into a Spark DataFrame.
2. **Split the dataset** into training and testing sets.
3. **Select a classification algorithm** using Decision Trees from Spark MLlib.
4. **Employ techniques** such as cross-validation and grid search to fine-tune the hyperparameters.
5. **Evaluate the performance** of the tuned model using relevant evaluation metrics.
6. **Generate predictions** on the testing data.
7. **Conduct a comparative analysis** between the predicted labels and the actual labels to assess the model's performance.

## Contact Information
For any questions or clarifications, please contact:

- Name: Aishah Zulkifli
- Email: aishahzulkifli20@gmail.com
