# Databricks notebook source
# MAGIC %md
# MAGIC # Train Custom Model in a SRC File
# MAGIC

# COMMAND ----------

import mlflow
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import src.custom_module

# COMMAND ----------

import pandas as pd

from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

import inspect
import sys
import os

sys.path.append(os.path.abspath(".."))

from src.custom_module import SklearnModelWrapper

# COMMAND ----------

print(mlflow.__version__)

# COMMAND ----------

# DBTITLE 1,Load dataset
raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )

def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    """Rename columns to be compatible with Feature Store"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# Run functions
renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# Drop target column ('quality') as it is not included in the feature table
features_df = df.drop('quality')
display(features_df)


# COMMAND ----------

# DBTITLE 1,Create a new database
database_name = "morgan_wine_db"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
spark.sql(f"USE DATABASE {database_name}")
# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"wine_quality"
print(table_name)

# COMMAND ----------

# DBTITLE 1,Create the feature table
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="wine features"
)

# COMMAND ----------

# MAGIC %md ## Train a model with feature store

# COMMAND ----------

## inference_data_df includes wine_id (primary key), quality (prediction target), and a real time feature
inference_data_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# Create the train and test datasets
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

X_test.head()

# COMMAND ----------

# DBTITLE 1,getfile code path
code_path = inspect.getfile(SklearnModelWrapper)

# COMMAND ----------

mlflow.sklearn.autolog(log_models=False) # Disable MLflow autologging and instead log the model using Feature Store


with mlflow.start_run() as run:
    rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)

    wrapper= SklearnModelWrapper(rf)
    mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=wrapper,
    registered_model_name="morgan_wine_model",
    input_example=X_test,
    code_path=[code_path]
    )
    y_pred = wrapper.predict(None, X_test)

    fs.log_model(
    artifact_path="model",
    model=wrapper,
    flavor=mlflow.pyfunc,
    registered_model_name="morgan_wine_model",
    training_set=training_set,
    code_path=["src/"]
    )


# COMMAND ----------

# DBTITLE 1,Batch Scoring
## For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("quality") # Drop the label column

predictions_df = fs.score_batch("models:/morgan_wine_model/latest", batch_input_df)
                                  
display(predictions_df["wine_id", "prediction"])
