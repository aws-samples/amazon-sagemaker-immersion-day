# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import re
import sys
import argparse
import pandas as pd
from optimum.neuron import version
from optimum.exporters.tasks import TasksManager
from optimum.exporters.neuron.model_configs import *
from optimum.neuron.distributed.parallelizers_manager import ParallelizersManager
from optimum.neuron.utils.training_utils import (
    _SUPPORTED_MODEL_NAMES,
    _SUPPORTED_MODEL_TYPES,
    _generate_supported_model_class_names
)

def training_models():
    # retrieve supported models for Tensor Parallelism
    tp_support = list(ParallelizersManager._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS.keys())

    # build compability table for training
    data_training = {'Model': []}
    for m in _SUPPORTED_MODEL_TYPES:
        if type(m) != str: m = m[0]
        if m=='gpt-2': m='gpt2' # fix the name
        model_id = len(data_training['Model'])
        model_link = f'<a rel="noopener noreferrer" target="_new" href="https://huggingface.co/models?sort=trending&search={m}">{m}</a>'
        data_training['Model'].append(f"{model_link} <font style='color: red;'><b>[TP]</b></font>" if m in tp_support else model_link)
        tasks = [re.sub(r'.+For(.+)', r'\1', t) for t in set(_generate_supported_model_class_names(m)) if not t.endswith('Model')]
        for t in tasks:
            if data_training.get(t) is None: data_training[t] = [''] * len(_SUPPORTED_MODEL_TYPES)
            data_training[t][model_id] = f'<a rel="noopener noreferrer" target="_new" href="https://huggingface.co/docs/transformers/model_doc/{m}#transformers.{m.title()}For{t}">doc</a>'        
    df_training = pd.DataFrame.from_dict(data_training).set_index('Model')
    return df_training.to_markdown()
    
def inference_models():
    # retrieve supported models for Tensor Parallelism
    tp_support = list(ParallelizersManager._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS.keys())

    # build compability table for inference
    meta = [(k,list(v['neuron'].keys())) for k,v in TasksManager._SUPPORTED_MODEL_TYPE.items() if v.get('neuron') is not None]
    data_inference = {'Model': []}
    for m,t in meta:
        model_id = len(data_inference['Model'])
        model_link = f'<a rel="noopener noreferrer" target="_new" href="https://huggingface.co/models?sort=trending&search={m}">{m}</a>'
        data_inference['Model'].append(f"{model_link} <font style='color: red;'><b>[TP]</b></font>" if m in tp_support else model_link)
        for task in t:
            if data_inference.get(task) is None: data_inference[task] = [''] * len(meta)
            data_inference[task][model_id] = f'<a rel="noopener noreferrer" target="_new" href="https://huggingface.co/models?pipeline_tag={task}&sort=trending&search={m}">doc</a>'

    df_inference = pd.DataFrame.from_dict(data_inference).set_index('Model')
    return df_inference.to_markdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input parameters of this script
    parser.add_argument("--output_file", type=str, required=True)
    
    try:
        args, _ = parser.parse_known_args()
        print(f"Dumping the metadata file to: {args.output_file}")
        with open(args.output_file, 'w') as f:
            f.write("# HF Optimum Neuron - Supported Models\n")
            f.write(f"**version: {version.__version__}**  \n")
            f.write("Models marked with <font style='color: red;'><b>[TP]</b></font> support **Tensor Parallelism** for training and inference\n")
            f.write("## Models/tasks for training\n")
            f.write(f"{training_models()}\n")
            f.write("## Models/tasks for inference\n")
            f.write(f"{inference_models()}\n")
    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)
        
    finally:
        print("Done! ", sys.exc_info())
        sys.exit(0)
