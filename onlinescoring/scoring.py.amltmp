# Import required libraries
import pandas as pd
from azureml.core.model import Model
from azureml.core import Workspace
import os
import logging
import json
import numpy
import joblib
import sklearn

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "PM2_MultiGrade_STFImodel.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()
    
# Load the workspace and the deployed model
# ws = Workspace.from_config()
# model = Model(ws, "deployed_model_name")

# # Load the input data as a Pandas DataFrame
# input_data = pd.read_csv("path/to/input/data.csv")

# # Use the deployed model to make predictions on the input data
# predictions = model.predict(input_data)

# # Save the predictions as a JSON file
# output_data = {"predictions": predictions}
# with open("path/to/output/data.json", "w") as f:
#     f.write(json.dumps(output_data))
