import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def convert_json_to_parquet():
    # Read the JSON file
    input_file = "data/tau2/domains/airline/tasks.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    # we use serialized now to avoid some null values got inserted and destroy the original semantics
    serialized_data = [{"id": task["id"], "serialized": json.dumps(task)} for task in data]

    print(f"Loaded {len(serialized_data)} records from {input_file}")

    # Convert to DataFrame
    df = pd.DataFrame(serialized_data)

    # Split into train and test (50/50)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")

    # Create output directory if it doesn't exist
    output_dir = "data/tau2/domains/airline"
    os.makedirs(output_dir, exist_ok=True)

    # Save as parquet files
    train_output = os.path.join(output_dir, "tasks_train_serialized.parquet")
    test_output = os.path.join(output_dir, "tasks_test_serialized.parquet")

    train_df.to_parquet(train_output, index=False)
    test_df.to_parquet(test_output, index=False)

    print(f"Saved training data to: {train_output}")
    print(f"Saved test data to: {test_output}")

    return train_output, test_output


if __name__ == "__main__":
    convert_json_to_parquet()
