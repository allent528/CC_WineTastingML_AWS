FROM python:3.9-slim

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Install PySpark and Numpy
RUN pip install pyspark==3.5.1 numpy

# Set working directory
WORKDIR /app

# Copy prediction script
COPY predict.py /app/predict.py

# Copy trained model (User must download this to local dir first)
COPY wine_quality_model /app/wine_quality_model

# Copy validation dataset for easy testing
COPY docs/ValidationDataset.csv /app/ValidationDataset.csv

# Set entrypoint
# Arguments passed to 'docker run' will be appended to this
ENTRYPOINT ["spark-submit", "predict.py"]
