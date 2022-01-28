 
import datetime
import json
import random
import boto3

STREAM_NAME = "ExampleInputStream"


def get_data():
    return random.choice(["1,1.44,-73.967393,40.756458,1,-73.98366,40.745642,0.5,0.5,0.0,0.0,7.5,300.0,0.0,1.0",
    "1,0.9,-73.948646,40.773943,1,-73.959834,40.76944,0.5,0.5,0.0,0.0,6.0,240.00000000000003,1.0,0.0",
    "2,1.5,-73.98050400000001,40.783272,1,-73.963669,40.794529,0.5,0.5,0.0,0.0,7.5,360.0,1.0,0.0",
    "2,13.6,-73.98812,40.748923,1,-73.90385500000001,40.887425,0.5,0.5,0.0,0.0,38.5,1320.0,1.0,0.0"])


def generate(stream_name, kinesis_client):
    while True:
        data = get_data()
        print(data)
        kinesis_client.put_record(
            StreamName=stream_name,
            Data=json.dumps(data),
            PartitionKey="partitionkey")


if __name__ == '__main__':
    generate(STREAM_NAME, boto3.client('kinesis'))
