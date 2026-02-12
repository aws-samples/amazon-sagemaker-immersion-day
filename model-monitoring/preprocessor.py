import base64
import csv
import io


# Column names from training-dataset-with-header.csv (excluding the label "Churn")
FEATURE_NAMES = [
    "Account Length", "VMail Message", "Day Mins", "Day Calls", "Eve Mins",
    "Eve Calls", "Night Mins", "Night Calls", "Intl Mins", "Intl Calls",
    "CustServ Calls", "State_AK", "State_AL", "State_AR", "State_AZ",
    "State_CA", "State_CO", "State_CT", "State_DC", "State_DE", "State_FL",
    "State_GA", "State_HI", "State_IA", "State_ID", "State_IL", "State_IN",
    "State_KS", "State_KY", "State_LA", "State_MA", "State_MD", "State_ME",
    "State_MI", "State_MN", "State_MO", "State_MS", "State_MT", "State_NC",
    "State_ND", "State_NE", "State_NH", "State_NJ", "State_NM", "State_NV",
    "State_NY", "State_OH", "State_OK", "State_OR", "State_PA", "State_RI",
    "State_SC", "State_SD", "State_TN", "State_TX", "State_UT", "State_VA",
    "State_VT", "State_WA", "State_WI", "State_WV", "State_WY",
    "Area Code_408", "Area Code_415", "Area Code_510",
    "Int'l Plan_no", "Int'l Plan_yes", "VMail Plan_no", "VMail Plan_yes",
]


def _decode(data, encoding):
    if encoding == "BASE64":
        return base64.b64decode(data).decode("utf-8")
    return data


def preprocess_handler(inference_record):
    input_data = inference_record.endpoint_input.data
    input_encoding = inference_record.endpoint_input.encoding

    input_csv = _decode(input_data, input_encoding)
    values = next(csv.reader(io.StringIO(input_csv.strip())))

    result = {name: float(val) for name, val in zip(FEATURE_NAMES, values)}

    output_data = inference_record.endpoint_output.data
    output_encoding = inference_record.endpoint_output.encoding
    output_csv = _decode(output_data, output_encoding)
    result["prediction"] = float(output_csv.strip())

    return result
