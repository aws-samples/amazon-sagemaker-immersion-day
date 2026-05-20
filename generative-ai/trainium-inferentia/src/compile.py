# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
os.environ['NEURON_RT_NUM_CORES'] = '1'
import sys
import glob
import json
import torch
import shutil
import tarfile
import logging
import argparse
import traceback
import optimum.neuron
from transformers import AutoTokenizer

def model_fn(model_dir, context=None):
    task = os.environ.get("TASK")
    if task is None: raise Exception("Invalid TASK. You need to invoke the compilation job once to set TASK variable")
        
    NeuronModel = eval(f"optimum.neuron.NeuronModelFor{task}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = NeuronModel.from_pretrained(model_dir)
    return model,tokenizer

def input_fn(input_data, content_type, context=None):
    if content_type == 'application/json':
        req = json.loads(input_data)
        prompt = req.get('prompt')
        if prompt is None or len(prompt) < 3:
            raise("Invalid prompt. Provide an input like: {'prompt': 'text text text'}")
        return prompt
    else:
        raise Exception(f"Unsupported mime type: {content_type}. Supported: application/json")    

def predict_fn(input_object, model_tokenizer, context=None):
    model,tokenizer = model_tokenizer
    inputs = tokenizer(input_object, truncation=True, return_tensors="pt")
    logits = model(**inputs).logits
    idx = logits.argmax(1, keepdim=True)
    conf = torch.gather(logits, 1, idx)
    return torch.cat([idx,conf], 1)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.    
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--dynamic_batch_size", type=bool, default=False)
    parser.add_argument("--input_shapes", type=str, required=True)
    parser.add_argument("--is_model_compressed", type=bool, default=True)
    
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])    
    parser.add_argument("--checkpoint_dir", type=str, default=os.environ["SM_CHANNEL_CHECKPOINT"])
    
    args, _ = parser.parse_known_args()

    # Set up logging        
    logging.basicConfig(
        level=logging.getLevelName("DEBUG"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    NeuronModel = eval(f"optimum.neuron.NeuronModel{'For' + args.task if len(args.task) > 0 else ''}")
    logger.info(f"Checkpoint files: {os.listdir(args.checkpoint_dir)}")

    model_path = args.checkpoint_dir
    if args.is_model_compressed:
        logger.info("Decompressing model file...")
        with tarfile.open(os.path.join(args.checkpoint_dir, "model.tar.gz"), 'r:gz') as tar:
            tar.extractall(os.path.join(args.checkpoint_dir, "model"))
        model_path = os.path.join(args.checkpoint_dir, "model")
        logger.info(f"Done! Model path: {model_path}")
        logger.info(f"Model path files: {os.listdir(model_path)}")

    input_shapes = json.loads(args.input_shapes)
    model = NeuronModel.from_pretrained(model_path, export=True, dynamic_batch_size=args.dynamic_batch_size, **input_shapes)
    model.save_pretrained(args.model_dir)

    code_path = os.path.join(args.model_dir, 'code')
    os.makedirs(code_path, exist_ok=True)

    shutil.copy(__file__, os.path.join(code_path, "inference.py"))
    shutil.copy('requirements.txt', os.path.join(code_path, 'requirements.txt'))
