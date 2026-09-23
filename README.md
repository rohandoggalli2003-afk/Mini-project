for running the ml model 
 (venv) PS D:\AI-Connected-Vehicle-Analytics> python -m ml.preprocessing
(venv) PS D:\AI-Connected-Vehicle-Analytics> python -m ml.train_xgboost

deactivate (for venv deactivation) 


.\venv\Scripts\Activate.ps1 --- > for the activation

Terminal 1 — Start Kafka

cd D:\bigdata\kafka
Then:

.\bin\windows\kafka-server-start.bat .\config\kraft\server.properties
Keep this terminal open.

Terminal 2 — Start Producer
cd D:\AI-Connected-Vehicle-Analytics
Activate your virtual environment:

.\venv\Scripts\Activate.ps1
Then:

python -m kafka_stream.producer
Keep it running.

Terminal 3 — Start Consumer
Open another PowerShell:

cd D:\AI-Connected-Vehicle-Analytics
Activate environment:

.\venv\Scripts\Activate.ps1
Then:

python -m kafka_stream.consumer



for pyspark 

(venv) PS D:\AI-Connected-Vehicle-Analytics>


$env:SPARK_HOME="D:\bigdata\spark\spark-3.5.9-bin-hadoop3\spark-3.5.9-bin-hadoop3"

$env:PYSPARK_PYTHON="python"

$env:PYTHONPATH="$env:SPARK_HOME\python;$env:SPARK_HOME\python\lib\py4j-0.10.9.7-src.zip"

python -c "import pyspark; print('PySpark version:', pyspark.__version__)"


for ml model

PS D:\AI-Connected-Vehicle-Analytics> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\AI-Connected-Vehicle-Analytics\venv\Scripts\Activate.ps1)
(venv) PS D:\AI-Connected-Vehicle-Analytics> python -m ml.preprocessing
(venv) PS D:\AI-Connected-Vehicle-Analytics> python -m ml.train_xgboost
