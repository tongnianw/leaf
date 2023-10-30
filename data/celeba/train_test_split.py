import json
import os
from sklearn.model_selection import train_test_split

LABEL = 'Smiling'
SEED = 1
ALPHA = 0.5


def main():
    # Specify the input directory where the client JSON files are saved
    input_directory = f"/home/tongnian/leaf/data/celeba/fairness_data/{LABEL}_seed_{SEED}_dir_{ALPHA}"

    # Create a directory to save the split datasets
    output_directory = f"/home/tongnian/CelebA_fairness/data/{LABEL}_seed_{SEED}_dir_{ALPHA}"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List all JSON files in the input directory
    client_json_files = [f for f in os.listdir(input_directory) if f.endswith(".json")]

    # Process each client's JSON file
    for i, client_json_file in enumerate(client_json_files):
        # Create a subdirectory for the client
        client_dir = os.path.join(output_directory, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)

        # Read the client's JSON data
        with open(os.path.join(input_directory, client_json_file), "r") as json_file:
            client_data = [json.loads(line) for line in json_file]

        # Split the data into train, val, and test sets
        train_data, test_data = train_test_split(client_data, shuffle=True, test_size=0.4, random_state=SEED)
        # val_data, test_data = train_test_split(temp_data, shuffle=True, test_size=0.5, random_state=1)

        # Define file paths for train, val, and test datasets
        train_file_path = os.path.join(client_dir, "train.json")
        # val_file_path = os.path.join(client_dir, "val.json")
        test_file_path = os.path.join(client_dir, "test.json")

        # Save the split datasets into their respective folders
        with open(train_file_path, "w") as train_file:
            for data in train_data:
                json.dump(data, train_file)
                train_file.write('\n')  # Add a newline to separate entries

        # with open(val_file_path, "w") as val_file:
        #     for data in val_data:
        #         json.dump(data, val_file)
        #         val_file.write('\n')  # Add a newline to separate entries

        with open(test_file_path, "w") as test_file:
            for data in test_data:
                json.dump(data, test_file)
                test_file.write('\n')  # Add a newline to separate entries

        print(f"Client {i} dataset split and saved to {client_dir}")


if __name__ == '__main__':

    main()