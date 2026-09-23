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
