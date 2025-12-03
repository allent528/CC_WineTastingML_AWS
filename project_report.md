# Wine Quality Prediction Project Report

**Name:** Allen Thomas
**Date:** 12/2/2025

## Links
- **GitHub Repository:** https://github.com/allent528/CC_WineTastingML_AWS
- **Docker Hub Image:** https://hub.docker.com/r/allent528/wine-prediction

## Setup and Execution Instructions

### 1. Cloud Environment Setup (AWS)
1.  **S3 Bucket**: Ensure an S3 bucket (e.g., `s3winetasting`) exists and contains `TrainingDataset.csv` and `ValidationDataset.csv`.
2.  **EMR Cluster**:
    -   Create an EMR cluster with Spark installed.
    -   Ensure the cluster has access to the S3 bucket (IAM roles).
    -   SSH into the master node.

### 2. Parallel Model Training (Spark on AWS)
1.  Copy `train_model.py` to the EMR master node (or host it on S3).
2.  Submit the Spark job:
    ```bash
    spark-submit --master yarn --deploy-mode cluster train_model.py
    ```
    *Note: You may need to adjust `--master` depending on your setup (e.g., `local[*]` for testing).*
3.  **Process**: The script will perform a grid search over regularization parameters (`regParam`: 0.0, 0.01, 0.1, 0.3) and elastic net mixing (`elasticNetParam`: 0.0, 0.5, 1.0), evaluating each on the validation set.
4.  **Output**: The best performing model (highest F1 score) will be saved to `s3://wines3bucket/wine_quality_model`.

### 3. Prediction Application (Single EC2 Instance)
**Prerequisites**: Java and PySpark installed on the EC2 instance.
1.  Copy `predict.py` to the EC2 instance.
2.  Download the `ValidationDataset.csv` (or use a test file) to the local filesystem.
3.  Run the prediction application:
    ```bash
    spark-submit predict.py ValidationDataset.csv
    ```
4.  **Output**: The application will print the F1 score.

4.  **Dockerized Prediction Application**
**Prerequisites**: Docker installed.
1.  **Build the Image**:
    ```bash
    docker build -t wine-prediction .
    ```
2.  **Run the Container**:
    The container includes the model and validation dataset, so you can run it directly:
    ```bash
    docker run wine-prediction /app/ValidationDataset.csv
    ```
    *Alternatively, to test with a different file on your host machine:*
    ```bash
    docker run -v $(pwd):/data wine-prediction /data/YourFile.csv
    ```
    *(Note: On Windows Git Bash, you may need to use `//data/YourFile.csv` to avoid path conversion issues).*

**Results**:
-   **F1 Score**: 0.5718 (achieved on ValidationDataset.csv)

## ChatGPT/AI Copilot Usage Report
**Tool Used**: Google Deepmind's Antigravity Agent (acting as an AI Pair Programmer).

**Usage Description**:
-   **Code Generation**: The AI agent generated the initial boilerplate for `train_model.py` and `predict.py`, including SparkSession initialization, data loading with schema options, and the MLlib pipeline (VectorAssembler, LogisticRegression).
-   **Dockerization**: The AI agent created the `Dockerfile` to containerize the PySpark application, ensuring Java and Python dependencies were met.
-   **Refinement**: I reviewed the generated code, specifically checking the S3 paths and column handling. The AI correctly identified the need to handle quoted headers in the CSV files.

**Experience**:
The AI copilot significantly accelerated the development process by providing syntactically correct PySpark code and a working Docker configuration. It handled the boilerplate effectively, allowing me to focus on the logic and AWS configuration.
