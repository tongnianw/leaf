import os
import json
import random
import numpy as np
import pandas as pd

LABEL = 'Smiling'
SEED = 1
ALPHA = 0.5
MODE = 'write' # 'split', 'write'

random.seed(SEED)
np.random.seed(SEED)


def split_dataset(X_list, y_list, s_list, num_parties):

    X_array = np.array(X_list)
    y_array = np.array(y_list)
    s_array = np.array(s_list)

    # Generate sample counts among parties from a Dirichlet distribution
    dataidx_map = {}

    alpha = ALPHA  ########## adjust this to control heterogeneity ##########

    num_classes = 2
    num_groups = 2
    min_size = 0 # track minimal samples per user
    N = len(X_array)

    X = [[] for _ in range(num_parties)]
    y = [[] for _ in range(num_parties)]
    s = [[] for _ in range(num_parties)]

    idx_batch = [[] for _ in range(num_parties)]
    sum_size = 0

    for k in range(num_classes):
        for n in range(num_groups):
            print(f'label={k} group={n}')
            idx_k = np.where((y_array == k) & (s_array == n))[0]
            print(idx_k)
            np.random.shuffle(idx_k)
            print(idx_k)
            print('len(idx_k): ', len(idx_k))
            proportions = np.random.dirichlet(np.repeat(alpha, num_parties))
            proportions = np.array([p * (len(idx_j) < N/num_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            print('proportions 3:',proportions)
            
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            print('proportions 4:',proportions)

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            idx_length = [len(idx_j) for idx_j in idx_batch]
            print(idx_length)
            min_size = min(idx_length)
            print(min_size)
            sum_size = sum(idx_length)
            print(sum_size)
            print('-----------------------')

    for j in range(num_parties):
        # np.random.shuffle(idx_batch[j])
        dataidx_map[j] = idx_batch[j]

    print(len(dataidx_map))

    y_stat = [[] for _ in range(num_parties)]
    s_stat = [[] for _ in range(num_parties)]
    # assign data
    for client in range(num_parties):
        idxs = dataidx_map[client]
        X[client] = X_array[idxs]
        y[client] = y_array[idxs]
        s[client] = s_array[idxs]

        for i in np.unique(y[client]):
            y_stat[client].append((int(i), int(sum(y[client]==i))))

        for i in np.unique(s[client]):
            s_stat[client].append((int(i), int(sum(s[client]==i))))

    for client in range(num_parties):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: {np.unique(y[client])}\t Groups: {np.unique(s[client])}")
        print(f"\t\t Samples of labels: ", [i for i in y_stat[client]])
        print(f"\t\t Samples of groups: ", [i for i in s_stat[client]])
        print("-" * 70)

    return X, y, s



def main():

    num_parties = 10
    # Define a directory to save data files
    output_directory = f"/home/tongnian/leaf/data/celeba/fairness_data/{LABEL}_seed_{SEED}_dir_{ALPHA}/"
    os.makedirs(output_directory, exist_ok=True)

    if MODE == 'split':

        file_path = "/home/tongnian/leaf/data/celeba/data/raw/list_attr_celeba.txt"
        # Initialize empty lists to store column names and data
        columns = []
        data = []
        all_data = []

        # Open and read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract the column names (first row)
        columns = lines[1].split()

        # Process the data rows (skip the first useless row)
        for line in lines[2:]:
            # Split each line into values
            values = line.split()
            # Append values to a list named data
            all_data.append(values)

        # Now, 'columns' contains the column names, and 'data' contains the data as a list of lists
        print(columns[0]) # Image (X) - [0], Male (s) - [20], Young (y) - [-1]

        # Randomly select 20,000 data samples from 'data'
        data = random.sample(all_data, 20000)

        # Convert elements from strings to integers
        data = [[row[0]] + [int(value) for value in row[1:]] for row in data]

        # Replace all occurrences of -1 with 0 in the list
        # Use a nested list comprehension to iterate through the elements
        data = [[0 if value == -1 else value for value in row] for row in data]
        print('whole data length: ', len(data))

        image_column_index = 0 # image name - X
        # Extract all elements from the column and get a list
        X_list = [row[image_column_index] for row in data]

        group_column_index = 20 # Male - s
        # Extract all elements from the group column and get a list
        s_list = [row[group_column_index] for row in data]

        ########################################################
        # target_column_index = -1 # Young - y
        # # Extract all elements from the column and get a list
        # y_list = [row[target_column_index] for row in data]

        target_column_index = 31 # Smiling - y
        # Extract all elements from the column and get a list
        y_list = [row[target_column_index] for row in data]
        ########################################################

        # # Filter rows whose Young ([-1] column) value is equal to 1 --> filter classes
        # matching_rows = [row for row in data if int(row[-1]) == 1]
        # print(len(matching_rows))
        # matching_rows = [row for row in data if int(row[-1]) == 0]
        # print(len(matching_rows))

        X, y, s = split_dataset(X_list, y_list, s_list, num_parties)

        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        s = np.array(s, dtype=object)
        
        # Save the array as a .npy file
        np.save(output_directory + 'all_X.npy', X)
        np.save(output_directory + 'all_y.npy', y)
        np.save(output_directory + 'all_s.npy', s)


    elif MODE == 'write':
        # Load the array from the .npy file
        X = np.load(output_directory + 'all_X.npy', allow_pickle=True)
        y = np.load(output_directory + 'all_y.npy', allow_pickle=True)
        s = np.load(output_directory + 'all_s.npy', allow_pickle=True)

        for client_idx in range(num_parties):

            # Create JSON files for each client
            client_json_file = os.path.join(output_directory, f"client_{client_idx}.json")

            with open(client_json_file, "w") as json_file:
                # Iterate through the client's data and write each entry separately
                for img, label, group in zip(X[client_idx], y[client_idx], s[client_idx]):
                    # Data to be written
                    entry = {
                        "image": img.tolist(),
                        "y": label.tolist(),
                        "s": group.tolist(),
                    }

                    json.dump(entry, json_file)
                    json_file.write('\n')  # Add a newline to separate entries
            
        print('Done.')


if __name__ == '__main__':
	main()

