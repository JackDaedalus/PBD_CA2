# Databricks notebook source
# MAGIC %md
# MAGIC #  PBD - CA2
# MAGIC #  PROG9813: 2022/2023 - Programming for Big Data - Spark 
# MAGIC #  Assignment Two
# MAGIC 
# MAGIC #  Ciaran Finnegan – TU060 (Part Time) – Second Year 
# MAGIC #  MSc in Computer Science (Data Science)
# MAGIC 
# MAGIC #  Student No : D21124026 
# MAGIC #  April 2023

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Overview
# MAGIC ####  The dataset is from a fraud detection problem in Kaggle
# MAGIC 
# MAGIC The IEEE Computational Intelligence Society have compiled this dataset of online transactions which are both
# MAGIC credit and debit card purchases.
# MAGIC 
# MAGIC The Training set contains a flag to indicate which records were found to be fraudulent.
# MAGIC Using this data, the challenge is to build a fraud detection model and assess against a provided Test Set of unlabelled data.
# MAGIC 
# MAGIC Although each transaction is scored for likelihood of fraud, this is effectively a binary classification problem - marking 'unseen' 
# MAGIC records in the Test Data as 'Fraud' or 'Non-Fraud.
# MAGIC 
# MAGIC #### Source : https://www.kaggle.com/competitions/ieee-fraud-detection/overview 

# COMMAND ----------

# MAGIC %md
# MAGIC ####  1. Notebook Set Up

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Import Required Libraries

# COMMAND ----------

## Required PySpark Libraries
from pyspark.sql.functions import col, sum, count, when, isnull, udf
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, LinearSVC, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import ArrayType, DoubleType
from sklearn.metrics import confusion_matrix, roc_curve, auc



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Load Dataset 
# MAGIC Load Transaction data from Databriack file store
# MAGIC 
# MAGIC 
# MAGIC ##### Kaggle Fraud Transaction Training and Testing datasets are already uploaded into Databricks CE file store

# COMMAND ----------

# List the contents of the tables directory
file_list = dbutils.fs.ls('/FileStore/tables')

# Sort the files by name - the reverse sorting shows the Kaggle Train/Test data tables first
sorted_files = sorted(file_list, key=lambda x: x.name, reverse=True)

# Print the sorted file names, and file szie
for file_info in sorted_files:
  print(f"{file_info.name} \t\t- {file_info.size} bytes")


# COMMAND ----------

#file_location = "/FileStore/tables/train_transaction_reduced_M1Adj.csv"

##########################################################
## Load Transaction Data file - Training Set 
##########################################################

# File location and type of Kaggle online transaction data
file_location = "/FileStore/tables/train_transaction.csv"

# CSV options - Set Parameters for reading in data
file_type = "csv"
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_TrxnTrain = spark.read.format(file_type) \
              .option("inferSchema", infer_schema) \
              .option("header", first_row_is_header) \
              .option("sep", delimiter) \
              .load(file_location)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Explore Data
# MAGIC Step Two - Data exploration and preparation for Feature Engineering stage
# MAGIC 
# MAGIC 
# MAGIC ##### Display the primary characteristics of the data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Show Sample Layout
# MAGIC 
# MAGIC Show initial rows from the Training dataset from Kaggle

# COMMAND ----------

##########################################################
## Display Initial Rows in Training Set Data file 
##########################################################

display(df_TrxnTrain)

# COMMAND ----------

##########################################################
## Show Schema in Training Set Data file 
## This assists in a visual identification of the 
## categorical and numeric features in the data
##########################################################
df_TrxnTrain.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Show Sample Size
# MAGIC 
# MAGIC Show total numble of samples (rows) in the Training dataset from Kaggle

# COMMAND ----------

##  Display size of Training dataset - Caching to help with queries
print('Full Txrn data set size = ', df_TrxnTrain.cache().count())


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Check Data Balance
# MAGIC 
# MAGIC Fraud datasets are often heavily imbalanced with fraud records being a very low proportion of the total dataset

# COMMAND ----------

##########################################################
## Determine the balance of fraud and non-Fraud samples in
## the Kaggle Training dataset
##########################################################

# Group the data by the fraud indicator column and count the number of occurrences of each value
grouped_data = df_TrxnTrain.groupBy("isFraud").count().orderBy("count", ascending=False).collect()

# Create a bar chart trace
#trace = go.Bar(x=[row["isFraud"] for row in grouped_data], y=[row["count"] for row in grouped_data])

# Create a bar chart trace
trace = go.Bar(x=["Non-Fraud", "Fraud"], y=[row["count"] for row in grouped_data])


# Create the layout for the bar chart
layout = go.Layout(title="Bar Chart - Balance of Fraud and non-Fraud Records in Kaggle Training Dataset")

# Combine the trace and layout to create the figure
fig = go.Figure(data=[trace], layout=layout)

# Display the figure using the display function
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Identify Missing Data
# MAGIC 
# MAGIC Check the categorical and numeric data in the Kaggle Training set and identify missing/null values

# COMMAND ----------

###########################################################################
## Define the categorical features from the trxn training set to be encoded

## The Kaggle competition confirms the following categorical features in the Transaction data
## ProductCD
## card4 + card6
## P_emaildomain
## R_emaildomain
## M1 - M9

###########################################################################
categorical_trxn_features = ['ProductCD', 'card4', 'card6', 'P_emaildomain','R_emaildomain', 'M1', 'M2','M3','M4','M5','M6','M7','M8','M9']


# COMMAND ----------

###########################################################################
## Create a list of the numerical features
###########################################################################
num_trxn_features = [column for column in df_TrxnTrain.columns if column not in ["isFraud"] + categorical_trxn_features]


# COMMAND ----------

#display numercial features
num_trxn_features

# COMMAND ----------

# Using the created PySpark dataframe "df_TrxnTrain" to 
# create a new dataframe of Boolean values indicating null or not
null_trxn_counts = df_TrxnTrain.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_TrxnTrain.columns]) 


# COMMAND ----------

# Nulls in categorical Features in Kaggle dataset
display(null_trxn_counts[categorical_trxn_features]) 

# COMMAND ----------

# Nulls in numerical Features in Kaggle dataset
display(null_trxn_counts[num_trxn_features]) 

# COMMAND ----------

# Show scale of null values in numerical fields
df_nulltrxncount = null_trxn_counts[num_trxn_features] 
print('Training Null count set size = ', df_nulltrxncount.cache().count()) # Cache because accessing training data multiple times 


# COMMAND ----------

# Show scale of null values in numerical fields

# Set the threshold value
threshold = 1600

# Calculate the sum of each column
sums = df_nulltrxncount.select([sum(col(c)).alias(c) for c in df_nulltrxncount.columns])

# Filter out columns that have a total value that exceeds the threshold
exceeding_columns = [c for c in sums.columns if sums.select(col(c)).collect()[0][0] > threshold]

# Print the list of columns that exceed the threshold
print("Columns exceeding threshold:", exceeding_columns)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Prepare Data
# MAGIC Step Three - Data manipulation to prepare dataset for ML modellinh process
# MAGIC 
# MAGIC 
# MAGIC ##### Rework the dataset to to allow it successfully proceed onto the gneration of an accurate model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Balance Data
# MAGIC Feature Engineering - Balance out dataset to avoid bias in model against detecting fraud.
# MAGIC 
# MAGIC 
# MAGIC ##### The data contains a significant majority of records that are non fraudulent. Building a model with the original Kaggle dataset will
# MAGIC ##### generate a model that is biased towards identifying records as 'non-fraud'.
# MAGIC 
# MAGIC ##### This section of the workbook will down sample the non-fraud records so that there is a smaller dataset, but one with a 50/50 split 
# MAGIC ##### of fraud and non-fraud records. This should improve the accuracy of the final model by removing the biad towards the original majority lable.  

# COMMAND ----------

##########################################################
## Re-balance the number fraud and non-Fraud samples
##########################################################

# Count the number of samples in each class
counts = df_TrxnTrain.groupBy("isFraud").count().collect()
minority_class_count = counts[0][1]
majority_class_count = counts[1][1]


# Calculate the ratio of the number of samples in the minority class to the majority class
ratio = minority_class_count / majority_class_count

# Create a DataFrame with the minority class samples
minority_class = df_TrxnTrain.filter(df_TrxnTrain.isFraud == 1)

# Sample the majority class DataFrame to obtain a subsample with the same number of samples as the minority class DataFrame
majority_class = df_TrxnTrain.filter(df_TrxnTrain.isFraud == 0).sample(withReplacement=False, fraction=ratio, seed=42)

# Concatenate the minority class DataFrame and the balanced majority class DataFrame to obtain the balanced dataset
balanced_data = minority_class.union(majority_class)

# COMMAND ----------

##  Display size of Training dataset
##  Caching to help with queries
print('Full Txrn data set size = ', df_TrxnTrain.count())


##  Display size of Training dataset
##  Caching to help with queries
print('Re-balanced Txrn data set size = ', balanced_data.cache().count())

# COMMAND ----------

##########################################################
## Re-check the balance of fraud and non-Fraud samples 
##########################################################

# Group the data by the fraud indicator column and count the number of occurrences of each value
grouped_balanced_data = balanced_data.groupBy("isFraud").count().orderBy("count", ascending=False).collect()

# Create a bar chart trace
trace = go.Bar(x=["Non-Fraud", "Fraud"], y=[row["count"] for row in grouped_balanced_data])

# Create the layout for the bar chart
layout = go.Layout(title="Bar Chart - Re-Balance of Fraud and non-Fraud Records in Amended Training Dataset")

# Combine the trace and layout to create the figure
fig2 = go.Figure(data=[trace], layout=layout)

# Display the figure using the display function
display(fig2)

# COMMAND ----------

##########################################################
## Assign Balanced dataframe to become the dataset on 
## wich we will build the fraud detection model
##########################################################
df_TrxnTrain = balanced_data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Missing Data
# MAGIC Feature Engineering - To Handle missing data
# MAGIC 
# MAGIC 
# MAGIC ##### The data contains many null elements which must be addressed befor the model can be built

# COMMAND ----------

# Replace null values in string columns with empty strings
df_TrxnTrain = df_TrxnTrain.na.fill(" ", subset=categorical_trxn_features)

# COMMAND ----------

# Replace null values in numeric columns with zeros
df_TrxnTrain = df_TrxnTrain.na.fill(0, subset=num_trxn_features)

# COMMAND ----------

##########################################################
## Display Initial Rows in Training Set Data file 
##########################################################

display(df_TrxnTrain)

# COMMAND ----------

# Now check for number of NULLS

# assuming you already have a PySpark dataframe named "df_TrxnTrain"
# create a new dataframe of Boolean values indicating null or not
null_trxn_postprc_counts = df_TrxnTrain.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_TrxnTrain.columns]) 


# COMMAND ----------

# Nulls in categorical Features in Kaggle dataset
display(null_trxn_postprc_counts[categorical_trxn_features]) 



# COMMAND ----------

# Nulls in numerical Features in Kaggle dataset
display(null_trxn_postprc_counts[num_trxn_features]) 

# COMMAND ----------

# Show scale of null values in numerical fields
df_nulltrxncount_postprc = null_trxn_postprc_counts[num_trxn_features] 

print('Training Null count set size (Post Processing) = ', df_nulltrxncount_postprc.cache().count()) 
# Cache because accessing training data multiple times

# COMMAND ----------

# Show scale of null values in numerical fields

# Set the threshold value
threshold = 1600

# Calculate the sum of each column
sums = df_nulltrxncount_postprc.select([sum(col(c)).alias(c) for c in df_nulltrxncount_postprc.columns]) 

# Filter out columns that have a total value that exceeds the threshold
exceeding_columns = [c for c in sums.columns if sums.select(col(c)).collect()[0][0] > threshold] 

# Print the list of columns that exceed the threshold
print("Columns exceeding threshold:", exceeding_columns) 


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Split Data
# MAGIC Split Data into training/test sets
# MAGIC 
# MAGIC 
# MAGIC ##### The Kaggle competition provides a seperate 'Training Set' against which the user builds a fraud detections.
# MAGIC ##### This is the data just loaded into this workbook.
# MAGIC #
# MAGIC 
# MAGIC ##### This data is used to build and refine the competition model itself - based on 'master' Training data provided by Kaggle (ICS) - and hence the split now to build the model 
# MAGIC #
# MAGIC 
# MAGIC ##### The Kaggle competition provides a seperate 'Test Set' against which the user scores each transaction and submits an entry. This is loaded later in the Notebook when the model is built and refined.

# COMMAND ----------

###########################################################################
## Split data into train and test sets

###########################################################################

# A 70/30 split is chosen on the Training data from Kaggle
# The '42' value for the seeds ensures a repeatable split is used to allow effective comparisons of model performance

df_train_trxn_data, df_test_trxn_data = df_TrxnTrain.randomSplit([0.7, 0.3], seed=42)

print('Full Txrn data set size = ', df_TrxnTrain.count())
print('Training Txrn set size = ', df_train_trxn_data.cache().count(), '(', df_train_trxn_data.count()/df_TrxnTrain.count(), '%)') 
print('Test Txrn set size = ', df_test_trxn_data.cache().count(), '(', df_test_trxn_data.count()/df_TrxnTrain.count(), '%)')

# COMMAND ----------

###################################################################
# Verify that the proportion of labels is maintained in each split
###################################################################

# Define the label column name
label_col = "isFraud"

df_list = [df_TrxnTrain, df_train_trxn_data, df_test_trxn_data]
df_name = ['Original Kaggle','Train Split','Test Split']

name_index = 0


for df in df_list: 
  # Get the count of a column
  column_count = df.select("isFraud").count() 

  # Get the count of a certain value of a column
  fraud_value_count = df.filter(df.isFraud == 1).count()
  non_fraud_value_count = df.filter(df.isFraud == 0).count()
  
  print("Data Set... {}".format(df_name[name_index]))
  name_index += 1
  print("Total Label count", column_count)
  print("Fraud Count Ratio", fraud_value_count/column_count)
  print("Non Fraud Count Ratio", non_fraud_value_count/column_count)
  print("\n")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Categorical Data
# MAGIC Feature Engineering - To Handle Non-Numeric data
# MAGIC 
# MAGIC 
# MAGIC ##### The data contains a number of categorical features, which must be addressed befor the model can be built

# COMMAND ----------

# Additional Feature Engineering on KAGGLE Dataset

# Typically ML algorithms require data in Numeric format  (IT DEPENDS on language/API/Library/Tool you are using)


#    StringIndexer: Converts a column of string values to a column of label indexes. 
#    OneHotEncoder: Maps a column of category indices to a column of binary vectors, with at most one "1" in each 
#    row that indicates the category index for that row.

#    One-hot encoding in Spark is a two-step process. You first use the StringIndexer, followed by the OneHotEncoder.


# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
stringIndexer_tx = StringIndexer(inputCols=categorical_trxn_features, outputCols=[x + "Index" for x in categorical_trxn_features], handleInvalid="keep") 
encoder_tx = OneHotEncoder(inputCols=stringIndexer_tx.getOutputCols(), outputCols=[x + "OHE" for x in categorical_trxn_features]) 


# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Build Model
# MAGIC Step Four  - Steps to build as accurate a model as possible
# MAGIC 
# MAGIC 
# MAGIC ##### Set up the key steps to generate a pipeline of actions to build a model
# MAGIC ##### Assess the effectiveness of the model using evaluation metrics

# COMMAND ----------

#Background: Transformers, estimators, and pipelines
#
# Three important concepts in MLlib machine learning that are illustrated in this notebook are Transformers, Estimators, and Pipelines.
#
#    Transformer: Takes a DataFrame as input, and returns a new DataFrame. Transformers do not learn any parameters from the data and 
#    simply apply rule-based transformations to either prepare data for model training or generate predictions using a trained MLlib model. 
#    You call a transformer with .transform() method.
#
#    Estimator: Learns (or "fits") parameters from your DataFrame via a .fit() method and returns a Model, which is a transformer.
#
#    Pipeline: Combines multiple steps into a single workflow that can be easily run. Creating a machine learning model typically involves 
#    setting up many different steps and iterating over them. Pipelines help you automate this process.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Prepare Pipeline
# MAGIC Set up processing steps for Pipeline
# MAGIC 
# MAGIC 
# MAGIC ##### Set up the key steps to prepare the post feature engineered data for the processing steps necessary to build a model

# COMMAND ----------

# Kaggle Dataset

# Combine all feature columns into a single feature vector

# Most MLlib algorithms require a single features column as input. Each row in this column contains a vector of data points 
# corresponding to the set of features used for prediction.

# MLlib provides the VectorAssembler transformer to create a single vector column from a list of columns.

assemblerInputs_tx = [c + "OHE" for c in categorical_trxn_features] + num_trxn_features
vecAssembler_tx = VectorAssembler(inputCols=assemblerInputs_tx, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Train Basic Model
# MAGIC Model Training
# MAGIC 
# MAGIC 
# MAGIC ##### Begin with a Logisitcal Regression Model to build the transaction fraud detection model

# COMMAND ----------

####################################################################
## A user input value will determine which algoritm is used to build
## the model to generate the Kaggle competition submission
####################################################################

# Define a function to get the user input using the input() function
def get_numeric_input(prompt):
    try:
        value = float(input(prompt))
        return value
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return get_numeric_input(prompt)


# COMMAND ----------

# Use Databricks widgets to create a text input box for the user.
dbutils.widgets.text("user_input", "", "Enter a numeric value:")

# COMMAND ----------

# Use the get_numeric_input function to read the user input from the widget
user_value = dbutils.widgets.get("user_input")

# Validate and convert the user input to a float
iAlgoSelection = get_numeric_input(user_value)
print("Numeric value entered:", iAlgoSelection)

# COMMAND ----------

# Choose algorithm based on user input

if iAlgoSelection == 3:
  # Define the Model - using Gradient-Boosted Trees - GBT
  # GBT is another ensemble method that builds trees sequentially, learning from the previous tree's mistakes. 
  # GBTs often have high accuracy and can handle a mix of feature types.
  algo = GBTClassifier(featuresCol='features', labelCol='isFraud')
  print("GBTClassifier Algorithm selected:", iAlgoSelection)
elif iAlgoSelection == 2:
  # Define the Model - Random Forest - RF
  # Random Forest: An ensemble learning method that constructs multiple decision trees 
  # and aggregates their predictions by majority voting.
  algo = RandomForestClassifier(featuresCol="features", labelCol="isFraud", numTrees=100)
  print("RandomForestClassifier Algorithm selected:", iAlgoSelection)
else:  
   # Define the Model - using logistic regression
   # Logistic Regression: A linear method to model the probability of a certain class 
   # or event existing 
  algo = LogisticRegression(featuresCol="features", labelCol="isFraud", regParam=1.0)
  print("LogisticRegression Algorithm selected:", iAlgoSelection)


# COMMAND ----------

# Build Kaggle Pipeline

# A Pipeline is an ordered list of transformers and estimators. 
# Define a pipeline to automate and ensure repeatability of the transformations to be applied to a dataset. 
# The pipeline.fit() method returns a PipelineModel, which is a transformer.

# Define the pipeline based on the stages created in previous steps.
pipeline_tx = Pipeline(stages=[stringIndexer_tx, encoder_tx, vecAssembler_tx, algo])

# Define the pipeline model.
pipelineModel_tx = pipeline_tx.fit(df_train_trxn_data)


# COMMAND ----------

# Apply the pipeline model to the test dataset.
predDF_tx = pipelineModel_tx.transform(df_test_trxn_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Evaluate Basic Model
# MAGIC Refine Fraud Detection Model
# MAGIC 
# MAGIC 
# MAGIC ##### Display performance metrics for basic model

# COMMAND ----------

# Display the predictions from the chosen model. 
display(predDF_tx.select("features", "isFraud", "prediction", "probability"))


# COMMAND ----------

# Funtion to generate a graph showing model accuracy
def accGraphGen(predictions, algoType):
  # Extract Prediction Results
  labels_and_raw_predictions = predictions.select('isFraud', 'rawPrediction').rdd.collect()

  y_true = [x['isFraud'] for x in labels_and_raw_predictions]
  y_score = [x['rawPrediction'][1] for x in labels_and_raw_predictions]

  fpr, tpr, thresholds = roc_curve(y_true, y_score)
  roc_auc = auc(fpr, tpr)
  
  # Plot Graph
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) - ' + algoType + ' Model')
  plt.legend(loc="lower right")
  plt.show()

# COMMAND ----------

# Display a graphical depiction of prediction accuracy from the chosen model. 
if iAlgoSelection == 3:
  # Model Predictions - Gradient-Boosted Trees - GBT
  accGraphGen(predDF_tx, "Gradient-Boosted Trees")
elif iAlgoSelection == 2:
  # Model Predictions - Random Forest - RF
  accGraphGen(predDF_tx, "Random Forest")
else:  
   # Model Predictions  - Logistic Regression - LR
  accGraphGen(predDF_tx, "Logistic Regression")

# COMMAND ----------

# Evaluate the model - for KAGGLE competition
# Use the BinaryClassificationEvaluator to evalute the area under the ROC curve and MulticlassClassificationEvaluator to evalute the accuracy.

# AUPRC is a useful metric in scenarios where there is class imbalance or when the cost of false positives and false negatives are different. 
# It is more informative than the Area Under the Receiver Operating Characteristic (ROC) curve in such situations, as the AUPRC emphasizes 
# the performance of the model on the positive (minority) class.

bcEvaluator_roc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
bcEvaluator_pr = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderPR")
mcEvaluator = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="accuracy") # Maybe this should not be used?


print(f"Area under ROC curve TRXN: {bcEvaluator_roc.evaluate(predDF_tx)}")
print(f"Area under Precision-Recall curve TRXN: {bcEvaluator_pr.evaluate(predDF_tx)}")
print(f"Accuracy TRXN: {mcEvaluator.evaluate(predDF_tx)}")  # Try Binary Classification Evaluator for accuracy

# COMMAND ----------

# Deploy SQL to generate additional analysis on Fraud detection metrics
predDF_tx.createOrReplaceTempView("basicFraudPredictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT isFraud as Actual_Fraud, prediction as Predicted_Fraud, count(*) AS count
# MAGIC FROM basicFraudPredictions
# MAGIC GROUP BY isFraud, prediction

# COMMAND ----------

#############################################
## Function to Evaluate the Recall Metric
#############################################

## Recall = True Positives / (True Positives + False Negatives)

def computeRecall(predictions, label):
  # Compute True Positives (TP) and False Negatives (FN) counts
  metrics = (
      predictions
      .select(F.col(label).alias("actual"), F.col("prediction"))
      .groupBy("actual", "prediction")
      .count()
      .collect()
  )

  TP, FN = 0, 0
  for metric in metrics:
      actual, prediction, count = metric
      if actual == 1 and prediction == 1:
          TP = count
      elif actual == 1 and prediction == 0:
          FN = count

  # Calculate recall
  recall = TP / (TP + FN)
  print(f"Recall Value is : {recall:.4f}")


# COMMAND ----------

## Invoke function to calculate Recall value of model predictions
computeRecall(predDF_tx, "isFraud")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Refine Model
# MAGIC Refine Fraud Detection Model
# MAGIC 
# MAGIC 
# MAGIC ##### Use Hyperparameter Tuning and Cross Validation to refine fraud detection model

# COMMAND ----------

# Hyperparmeter Tuning of Kaggle Model
# MLlib provides methods to facilitate hyperparameter tuning and cross validation.
#
#    For hyperparameter tuning, ParamGridBuilder lets you define a grid search over a set of model hyperparameters.
#    For cross validation, CrossValidator lets you specify an estimator (the pipeline to apply to the input dataset), an evaluator, a grid space of 
#    hyperparameters, and the number of folds to use for cross validation.

# COMMAND ----------

# Set up HyperParameter Grid based on algorithm with which model 
# was generated
if iAlgoSelection == 3:
  # Model Predictions - Gradient-Boosted Trees - GBT
  paramGrid_tx = ParamGridBuilder() \
    .addGrid(algo.maxDepth, [3, 5]) \
    .addGrid(algo.maxIter, [10, 20]) \
    .build()
  
elif iAlgoSelection == 2:
  # CV Parameter Options  -  Random Forest - RF 
  # Build the RF parameter grid for tuning
  paramGrid_tx = ParamGridBuilder() \
    .addGrid(algo.numTrees, [50, 100]) \
    .addGrid(algo.maxDepth, [5, 10]) \
    .addGrid(algo.impurity, ["gini", "entropy"]) \
    .build()
  
else:  
   # CV Parameter Options  - Logistic Regression - LR
  paramGrid_tx = (ParamGridBuilder()
    .addGrid(algo.regParam, [0.1, 0.01])
    .addGrid(algo.elasticNetParam, [0, 1])
    .build())

# COMMAND ----------

# Create a 2-fold CrossValidator - the 3 fold option is causing a cluster timeout
cv_tx = CrossValidator(estimator=pipeline_tx, 
                       estimatorParamMaps=paramGrid_tx, 
                       evaluator=bcEvaluator_pr, 
                       numFolds=2)

# COMMAND ----------

###############################################################
## The Databricks CE edition will time out when processiing 
## Cross Validation on full 30K Training Set. 
## Hence the training datase for is reduced by 90% to avoid 
## failures during execution.
###############################################################

# Set the fraction for 8.5% cut
fraction = 0.085


# Set a random seed for reproducibility (optional)
seed = 42

# Create a new DataFrame with a 10% cut of the input DataFrame
dfReducedCVTrainSet = df_train_trxn_data.sample(withReplacement=False, fraction=fraction, seed=seed)

# Set the column name and the value you want to count
column_name = "isFraud"
non_fraud_value = 0
fraud_value = 1

# Filter the DataFrame based on the specific value and column
filtered_df_nonFraud = dfReducedCVTrainSet.filter(dfReducedCVTrainSet[column_name] == non_fraud_value)
filtered_df_Fraud = dfReducedCVTrainSet.filter(dfReducedCVTrainSet[column_name] == fraud_value)

# Count the number of entries with the specific value in the column
count_nonFraud = filtered_df_nonFraud.count()
count_Fraud = filtered_df_Fraud.count()

print(f"Number of non-Fraud entries '{non_fraud_value}' in column '{column_name}': {count_nonFraud}")
print(f"Number of Fraud entries '{fraud_value}' in column '{column_name}': {count_Fraud}")

# COMMAND ----------

# cvModel_tx = pipelineModel_tx
# cvModel_tx = cv_tx.fit(df_train_trxn_data) 

# Kaggle - Implement Cross Validation Build of Model

# Run cross validations. This step can be process intensive and returns 
# the best model found from the cross validation.
# CV executed on reduced Training Set to avoid processing timeout on Databricks CE
cvModel_tx = cv_tx.fit(dfReducedCVTrainSet) 


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Evaluate Refined Model
# MAGIC Evaluate Best Model After Tuning
# MAGIC 
# MAGIC 
# MAGIC ##### Determine How Effective the best post tuning model performs

# COMMAND ----------

# Evaluate model after Hyperparameter Tuning
# Use the model identified by the cross-validation to make predictions on the test dataset
cvPredDF_tx = cvModel_tx.transform(df_test_trxn_data)


# COMMAND ----------

# Evaluate the model's performance based on area under the ROC curve, Precions-Recall curve, and Accuracy 
print(f"Area under ROC curve: {bcEvaluator_roc.evaluate(cvPredDF_tx)}")
print(f"Area under Precision-Recall curve TRXN: {bcEvaluator_pr.evaluate(cvPredDF_tx)}")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF_tx)}")

# COMMAND ----------

# Display a graphical depiction of prediction accuracy from the chosen model after tuning. 
if iAlgoSelection == 3:
  # Model Predictions - Gradient-Boosted Trees - GBT
  accGraphGen(cvPredDF_tx, "Gradient-Boosted Trees")
elif iAlgoSelection == 2:
  # Model Predictions - Random Forest - SVM
  accGraphGen(cvPredDF_tx, "Random Forest")
else:  
   # Model Predictions  - Logistic Regression - LR
  accGraphGen(cvPredDF_tx, "Logistic Regression")

# COMMAND ----------

# Deploy SQL to generate additional analysis on Fraud detection metrics# Kaggle
cvPredDF_tx.createOrReplaceTempView("cvFraudPredictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT isFraud as Actual_Fraud, prediction as Predicted_Fraud, count(*) AS count
# MAGIC FROM cvFraudPredictions
# MAGIC GROUP BY isFraud, prediction

# COMMAND ----------

## Invoke function to calculate Recall value of model predictions
computeRecall(cvPredDF_tx, "isFraud")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Generate Kaggle Entry
# MAGIC Step Five - Use Kaggle Test Transactions to generate competition entry
# MAGIC 
# MAGIC 
# MAGIC ##### The Kaggle file 'Test Transactions' contains the unlablled transaction data against which the competition predictions must be generated.
# MAGIC   
# MAGIC ##### Using the model build in this Databricks notebook, a set of predictions for the 'Test Transactions' will be generated.  
# MAGIC   
# MAGIC ##### The prediction data will be converted into the format required for the Kaggle submission. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Read Kaggle Test Data
# MAGIC 
# MAGIC The Kaggle 'Test Transactions' file has already been loaded into Databricks

# COMMAND ----------

#file_locationCompTest = "/FileStore/tables/test_transaction_reduced.csv"

##########################################################
## Load Kaggle Test Data file - Competition Data
##########################################################

# File location and type of Kaggle online transaction data
file_locationCompTest = "/FileStore/tables/test_transaction.csv"


# The applied options are for CSV files. For other file types, these will be ignored.
df_KaggleCompTest = spark.read.format(file_type) \
                  .option("inferSchema", infer_schema) \
                  .option("header", first_row_is_header) \
                  .option("sep", delimiter) \
                  .load(file_locationCompTest)

# COMMAND ----------

##  Display size of Test dataset against which to build competition fraud scores
##  Caching to help with queries
print('Full Kagle Competition Test data set size = ', df_KaggleCompTest.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Process Kaggle Test Data
# MAGIC 
# MAGIC Remove null values from 'Test Transaction' data. 

# COMMAND ----------

# Replace null values in string columns with empty strings
df_KaggleCompTest = df_KaggleCompTest.na.fill(" ", subset=categorical_trxn_features)

# COMMAND ----------

# Replace null values in numeric columns with zeros
df_KaggleCompTest = df_KaggleCompTest.na.fill(0, subset=num_trxn_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Generate Kaggle Predictions
# MAGIC 
# MAGIC Use fraud detection model build in this notebook to scan each sample row and generate a prediction for whether the record should be marked as fraudulent or not.

# COMMAND ----------

# Evaluate model after Hyperparameter Tuning
# Use the model identified by the cross-validation to make predictions on the test dataset
cvPredDF_CompTest = cvModel_tx.transform(df_KaggleCompTest)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Extract Fraud Score 
# MAGIC 
# MAGIC The probability score in this fraud classification model are stored in a DenseVector object.
# MAGIC 
# MAGIC Extract the score that indicates liklihood of a fraudulent transaction.
# MAGIC 
# MAGIC Use these values in the generation of the Kaggle competition submission.

# COMMAND ----------

##########################################################
## Define a user-defined function (UDF) to convert the DenseVector object into an array of probabilities
##########################################################

def dense_vector_to_array(vector: DenseVector):
    return vector.toArray().tolist()

dense_vector_to_array_udf = udf(dense_vector_to_array, ArrayType(DoubleType()))


# COMMAND ----------

##########################################################
## Apply the UDF to your DataFrame containing the probability column
## The Probability score are extracted from vector into an array in 
## a new column 
##########################################################

# Assuming 'predictions' is your DataFrame with the 'probability' column
cvPredDF_CompTest_with_prob_array = cvPredDF_CompTest.withColumn("probability_array", dense_vector_to_array_udf(cvPredDF_CompTest["probability"]))

# Assuming 'predictions_with_prob_array' is your DataFrame with the 'probability_array' column
cvPredDF_CompTest_with_prob_array = cvPredDF_CompTest_with_prob_array.withColumn("Fraud_Score", col("probability_array").getItem(1))



# COMMAND ----------

##########################################################
## Show the prodiction vectors/arrays generated from the Kaggle Test data
##########################################################
display(cvPredDF_CompTest_with_prob_array.select("TransactionID", "probability", "probability_array", "Fraud_Score"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Format Kaggle Submission
# MAGIC 
# MAGIC Amend prediction output into format required for Kaggle competition.

# COMMAND ----------

#############################################
# Use SQL to analyse the submission data for the Kaggle competition
##############################################
cvPredDF_CompTest_with_prob_array.createOrReplaceTempView("CompetitionPredictions")

# COMMAND ----------

##########################################################
## Display the numbers of non-Fraud(0) and Fraud(1) 
## predicted in the Kaggle dataset
##########################################################

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT prediction as Predicted_Fraud, count(*) AS Txn_Count
# MAGIC FROM CompetitionPredictions
# MAGIC GROUP BY prediction

# COMMAND ----------

##########################################################
## Display a selection of the non-Fraud(0) and Fraud(1) 
## records with Fraud scores as predicted in the Kaggle dataset
##########################################################

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT TransactionID, prediction as Predicted_Fraud, Fraud_Score
# MAGIC FROM CompetitionPredictions
# MAGIC ORDER BY TransactionID

# COMMAND ----------

##########################################################
## Generate a dataframe to build a CSV export from this 
## notebook to create the Kaggle competition submission
##########################################################
df_KaggleSubmission = cvPredDF_CompTest_with_prob_array["TransactionID","Fraud_Score"]

# COMMAND ----------

display(df_KaggleSubmission)
