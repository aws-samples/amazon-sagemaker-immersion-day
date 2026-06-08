import json
import os
import numpy as np
import xgboost as xgb


def input_fn(input_data, content_type):
    if content_type == "application/json":
        obj = json.loads(input_data)
        array = np.array(obj)
        return xgb.DMatrix(array)
    elif content_type == "text/csv":
        data = []
        for line in input_data.strip().split("\n"):
            data.append([float(x) for x in line.split(",")])
        return xgb.DMatrix(np.array(data))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def model_fn(model_dir):
    model_file = os.path.join(model_dir, "xgboost-model")
    booster = xgb.Booster()
    booster.load_model(model_file)
    return booster
