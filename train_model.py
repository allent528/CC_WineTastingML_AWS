import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_header(df):
    """Removes extra quotes from column names."""
    for col_name in df.columns:
        new_name = col_name.replace('"', '')
        df = df.withColumnRenamed(col_name, new_name)
    return df

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("WineQualityPredictionTraining") \
        .getOrCreate()

    # S3 Paths
    # Note: Using s3a for S3 access. Ensure hadoop-aws dependencies are available if running locally.
    # On EMR, s3:// is usually supported natively.
    train_path = "s3://wines3bucket/TrainingDataset.csv"
    val_path = "s3://wines3bucket/ValidationDataset.csv"
    model_output_path = "s3://wines3bucket/wine_quality_model"

    print(f"Loading training data from {train_path}...")
    
    # Load Training Data
    # The file uses semicolons as delimiters and has quotes around headers
    train_df = spark.read.option("header", "true") \
        .option("sep", ";") \
        .option("inferSchema", "true") \
        .csv(train_path)
    
    train_df = clean_header(train_df)
    
    # Load Validation Data
    print(f"Loading validation data from {val_path}...")
    val_df = spark.read.option("header", "true") \
        .option("sep", ";") \
        .option("inferSchema", "true") \
        .csv(val_path)
    
    val_df = clean_header(val_df)

    # Feature Engineering
    # All columns except 'quality' are features
    feature_cols = [c for c in train_df.columns if c != 'quality']
    print(f"Feature columns: {feature_cols}")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    train_data = assembler.transform(train_df).select("features", "quality")
    val_data = assembler.transform(val_df).select("features", "quality")

    # Train Model with Hyperparameter Tuning
    # We will try different values for regularization and elastic net mixing
    print("Starting Hyperparameter Tuning...")
    
    # Define hyperparameter grid
    reg_params = [0.0, 0.01, 0.1, 0.3]
    elastic_net_params = [0.0, 0.5, 1.0]
    
    best_f1 = -1.0
    best_model = None
    best_params = {}
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="f1")

    for reg in reg_params:
        for en in elastic_net_params:
            print(f"Training with regParam={reg}, elasticNetParam={en}...")
            
            lr = LogisticRegression(
                labelCol="quality", 
                featuresCol="features", 
                family="multinomial",
                regParam=reg,
                elasticNetParam=en
            )
            
            # Train on training data
            model = lr.fit(train_data)
            
            # Evaluate on validation data
            predictions = model.transform(val_data)
            f1_score = evaluator.evaluate(predictions)
            
            print(f"  -> Validation F1 Score: {f1_score}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model
                best_params = {'regParam': reg, 'elasticNetParam': en}

    print("="*30)
    print(f"Best F1 Score: {best_f1}")
    print(f"Best Parameters: {best_params}")
    print("="*30)

    # Save Best Model
    print(f"Saving best model to {model_output_path}...")
    best_model.write().overwrite().save(model_output_path)
    
    spark.stop()

if __name__ == "__main__":
    main()
