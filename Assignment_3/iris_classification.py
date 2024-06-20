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
    StructField("sepal_length", FloatType(), True),
    StructField("sepal_width", FloatType(), True),
    StructField("petal_length", FloatType(), True),
    StructField("petal_width", FloatType(), True),
    StructField("species", StringType(), True)
])

# Load the dataset
iris_df = spark.read.csv("/user/maria_dev/Aishah/iris.csv", header=False, schema=schema)

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

