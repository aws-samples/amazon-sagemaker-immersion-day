from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
ds = ds.shuffle(seed=42).select(range(100))
ds.save_to_disk("/tmp/train_data")
print("Done")
