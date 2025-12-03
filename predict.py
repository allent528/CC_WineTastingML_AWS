import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def clean_header(df):
    """Removes extra quotes from column names."""
    for col_name in df.columns:
        new_name = col_name.replace('"', '')
        df = df.withColumnRenamed(col_name, new_name)
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: spark-submit predict.py <test_file_path>")
        sys.exit(1)

    test_file_path = sys.argv[1]

    # HACK: Fix for Git Bash on Windows converting paths to C:/... when running Docker
    # If we receive a Windows path inside our Linux container, it's definitely wrong.
    # We fall back to the bundled file to ensure it works for the user.
    if (test_file_path.startswith("C:/") or test_file_path.startswith("C:\\")):
        print(f"WARNING: Detected Windows path '{test_file_path}' inside Linux container.")
        print("Git Bash likely mangled the path argument. Using bundled /app/ValidationDataset.csv instead.")
        test_file_path = "/app/ValidationDataset.csv"
    
    # Initialize Spark Session
    # Suppress excessive logging
    spark = SparkSession.builder \
        .appName("WineQualityPrediction") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Model Path
    # We expect the model to be in the same directory or mounted
    model_path = "wine_quality_model"

    print(f"Loading model from {model_path}...")
    try:
        model = LogisticRegressionModel.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Loading test data from {test_file_path}...")
    # Load Test Data
    # Assuming same format as training data (semicolon delimiter, quoted headers)
    try:
        test_df = spark.read.option("header", "true") \
            .option("sep", ";") \
            .option("inferSchema", "true") \
            .csv(test_file_path)
        
        test_df = clean_header(test_df)
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # Feature Engineering
    # We need to assemble features just like in training
    # Columns: fixed acidity, volatile acidity, ..., alcohol
    # Target: quality
    feature_cols = [c for c in test_df.columns if c != 'quality']
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    try:
        test_data = assembler.transform(test_df).select("features", "quality")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        print("Ensure the test file has the correct columns.")
        sys.exit(1)

    # Make Predictions
    print("Running predictions...")
    predictions = model.transform(test_data)

    # Evaluate
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="f1")
    
    f1_score = evaluator.evaluate(predictions)
    
    print("="*30)
    print(f"F1 Score: {f1_score}")
    print("="*30)

    spark.stop()

if __name__ == "__main__":
    main()
