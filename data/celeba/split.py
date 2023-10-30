import os
import json
import random
import numpy as np

random.seed(1)
np.random.seed(1)

def split_dataset(dataset, users_group_dict, setting, num_parties):
    # Extract user and group information from the dataset
    users = dataset["users"]
    user_data = dataset["user_data"]
    num_samples = sum(dataset["num_samples"])
    num_users = len(users)
    print(num_users, num_samples) # 6057 10000
    users_group = np.array(users_group_dict['s'])
    print(users_group)

    # # Initialize lists to store user IDs for each subgroup
    # subgroup_users = {
    #     "s=0.0,y=0.0": [],
    #     "s=0.0,y=1.0": [],
    #     "s=1.0,y=0.0": [],
    #     "s=1.0,y=1.0": []
    # }
    # # Iterate through the JSON data and categorize users into subgroups
    # for user_id, user_data in dataset["user_data"].items():
    #     s_value = user_data["s"][0]
    #     y_value = user_data["y"][0]
    #     subgroup_key = f"s={s_value},y={y_value}"
    #     subgroup_users[subgroup_key].append(user_id)
    # # Convert the subgroup_users dictionary to a list of subgroups
    # subgroup_users_list = [[subgroup, user_ids] for subgroup, user_ids in subgroup_users.items()]

    if setting == 'i':
        # Generate sample counts among parties from a Dirichlet distribution
        dataidx_map = {}
        alpha = 1.0
        num_classes = 2
        min_size = 0 # track minimal samples per user

        X = [[] for _ in range(num_parties)]
        y = [[] for _ in range(num_parties)]
        s = [[] for _ in range(num_parties)]
        statistic = [[] for _ in range(num_parties)]

        idx_batch = [[] for _ in range(num_parties)]

        for k in range(num_classes):
            group_idx = np.where(users_group == float(k))[0]
            print(group_idx)
            idx_k = np.array([users_group_dict['users'][i] for i in group_idx])
            np.random.shuffle(idx_k)
            print(idx_k)
            print(len(idx_k))
            proportions = np.random.dirichlet(np.repeat(alpha, num_parties))
            # print('proportions 1:',len(proportions))
            # proportions = np.array([p*(len(idx_j)<num_users/num_parties) for p,idx_j in zip(proportions,idx_batch)])
            # print('proportions 2:',len(proportions))
            proportions = proportions/proportions.sum()
            # print(proportions)
            print('proportions 3:',len(proportions))
            
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)#[:-1]
            print(proportions)
            print('proportions 4:',len(proportions))

            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch, np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            print(min_size)
            print('-----------')

        for j in range(num_parties):
            # np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]

        print(len(dataidx_map))
        # assign data
        for client in range(num_parties):
            idxs = dataidx_map[client]
            # print('idxs:', idxs)
            X[client] = [dataset["user_data"][idxs[i]]["x"] for i in range(len(idxs))]
            y[client] = [dataset["user_data"][idxs[i]]["y"] for i in range(len(idxs))]
            s[client] = [dataset["user_data"][idxs[i]]["s"] for i in range(len(idxs))]

        # for client_idx in range(num_parties):
        #     X[client_idx] = [j for sub in X[client_idx] for j in sub]
        #     y[client_idx] = [j for sub in y[client_idx] for j in sub]
        #     s[client_idx] = [j for sub in s[client_idx] for j in sub]
    
        # for i in np.unique(y[client]):
        #     statistic[client].append(i, int(sum(y[client]==i)))

        # for client in range(num_parties):
        #     print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        #     print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        #     print("-" * 50)

        return X, y, s, dataidx_map

    # proportions = np.random.dirichlet(np.repeat(alpha, num_parties))
    # # Ensure that no proportion is zero
    # while any(prop == 0 for prop in proportions):
    #     proportions = np.random.dirichlet(np.repeat(alpha, num_parties))
    # print('proportions:', proportions)
    # print('proportions sum:', proportions.sum())

    # min_size = np.min(proportions * num_users)
    # user_counts = (proportions * num_users).astype(int)
    # print('dirichlet generate user counts for each party: ', user_counts)
    # print('total user counts: ', sum(user_counts))

    # # Adjust sample counts to ensure the total matches num_users
    # sample_counts_diff = num_users - sum(user_counts)
    # if sample_counts_diff > 0:
    #     print(f'distributing remaining {sample_counts_diff} users to random parties...')
    #     # If there are extra samples, distribute them to random parties
    #     for _ in range(sample_counts_diff):
    #         random_party = random.choice(range(num_parties))
    #         user_counts[random_party] += 1
    # print('final total user counts: ', user_counts)

    elif setting == 'ii':

        # Initialize data structures to store the distribution
        clients_data = [{} for _ in range(10)]

        # Identify the minority subgroup
        minority_group = {"y": 0.0, "s": 0.0}
        # Separate users into majority and minority groups
        minority_data = {}
        majority_data = {}
        
        # filtering the users into two groups: the "majority group" and the "minority group" 
        # based on the values of "y" (label) and "s" (group or sensitive attribute) in the dataset.
        for user_id, user_info in user_data.items():
            # (1) checks if the label ("y") of a user does not match the label of the "minority group"
            # (2) checks if the group or sensitive attribute ("s") of a user does not match the group of the "minority group"
            if user_info["y"][0] != minority_group["y"] or user_info["s"][0] != minority_group["s"]:
                for img_id, img_y, img_s in zip(user_info["x"], user_info["y"], user_info["s"]):
                    majority_data[img_id] = {"y": img_y, "s": img_s}
            else:
                for img_id, img_y, img_s in zip(user_info["x"], user_info["y"], user_info["s"]):
                    minority_data[img_id] = {"y": img_y, "s": img_s}
        
        # Shuffle the minority data (as before)
        minority_ids = list(minority_data.keys())
        random.shuffle(minority_ids)
        
        # Determine the number of parties for each ratio
        num_minority_parties = 5
        remaining_clients = num_parties - num_minority_parties

        # Calculate the number of minority data points per client
        total_minority_samples = len(minority_ids)
        minority_samples_to_select = int(0.8 * total_minority_samples)
        
        # Divide the minority data into two parts: 80% and 20%
        selected_minority_images = minority_ids[:minority_samples_to_select]
        remaining_minority_images = minority_ids[minority_samples_to_select:]

        # Divide the selected 80% minority data evenly among the first five clients
        # clients_data = [[] for _ in range(num_parties)]
        minority_samples_per_party = minority_samples_to_select // num_minority_parties
        for i in range(num_minority_parties):
            start_idx = i * minority_samples_per_party
            end_idx = start_idx + minority_samples_per_party
            selected_img_ids = selected_minority_images[start_idx:end_idx]
            # For each selected image, add its data to the corresponding client
            for img_id in selected_img_ids:
                clients_data[i][img_id] = minority_data[img_id]

        # Distribute remaining 20% minority data to the remaining clients
        minority_samples_per_remaining_client = len(remaining_minority_images) // remaining_clients
        for i in range(num_minority_parties, num_parties):
            start_idx = (i - num_minority_parties) * minority_samples_per_remaining_client
            end_idx = start_idx + minority_samples_per_remaining_client
            remaining_img_ids = remaining_minority_images[start_idx:end_idx]
            # For each remaining image, add its data to the corresponding client
            for img_id in remaining_img_ids:
                clients_data[i][img_id] = minority_data[img_id]

        # Divide the majority data evenly among all the clients
        majority_ids = list(majority_data.keys())
        majority_samples_per_client = len(majority_ids) // num_parties
        for i in range(num_parties):
            start_idx = i * majority_samples_per_client
            end_idx = start_idx + majority_samples_per_client
            majority_img_ids_client = majority_ids[start_idx:end_idx]
            # For each majority image, add its data to the corresponding client
            for img_id in majority_img_ids_client:
                clients_data[i][img_id] = majority_data[img_id]

        # Print the partitioned data for each client
        for i, client_data in enumerate(clients_data):
            client_id = f"client_{i}"
            print(client_id, ":", len(client_data))
            # print(client_id, ":", client_data)
        
        return clients_data



def main():

    # 1. Open the JSON file for reading
    with open('/home/tongnian/leaf/data/celeba/data/all_data/all_data.json', 'r') as file:
        # 2. Parse the JSON data using json.load()
        data = json.load(file)

    num_samples = data['num_samples']
    users = data['users']
    print(users[0])
    user_data = data['user_data']

    y = data['user_data']['1652']
    print(y)

    users_group_dict = {"users": [], "s": []}

    # Iterate through user_data
    for user_id, user_info in data["user_data"].items():
        # Extract "s" and "users" values
        s_value = user_info["s"][0]
        user_value = user_id

        # Append the values to the dictionary
        users_group_dict["s"].append(s_value)
        users_group_dict["users"].append(user_value)

    print(len(users_group_dict['s']))
    print(len(users_group_dict['users']))

    num_parties = 10
    setting = 'ii'

    if setting == 'ii':  # bias propogation setting ii
        clients_data = split_dataset(data, users_group_dict, setting, num_parties)
        
        # Define a directory to save the JSON files
        output_directory = "/home/tongnian/leaf/data/celeba/data/non-iid-ii/"

        # Create JSON files for each client
        for i, client_data in enumerate(clients_data):
            client_id = f"client_{i}"
            # print('client_data', client_data)

            # # Initialize client JSON data
            # client_json = {
            #     "person": [],
            #     "image": [],
            #     "target": [],
            #     "sens": []
            # }

            # # Populate client JSON data
            # for user_id in client_data:
            #     user_info = data["user_data"][user_id]
            #     client_json["person"].append(user_id)
            #     client_json["image"].append(user_info["x"])
            #     client_json["target"].append(user_info["y"])
            #     client_json["sens"].append(user_info["s"])

            # Save the client JSON to a file
            client_json_file = os.path.join(output_directory, f"{client_id}.json")
            with open(client_json_file, "w") as json_file:
                # Iterate through the client's data and write each entry separately
                for img_id, img_info in client_data.items():
                    # entry = {img_id: img_info}
                    entry = {"image": img_id, "y": img_info["y"], "s": img_info["s"]}
                    json.dump(entry, json_file)
                    json_file.write('\n')  # Add a newline to separate entries

            print(f"Client {i} JSON data saved to {client_json_file}")

    elif setting == 'i':
        # Split the dataset into parties with the specified ratio
        X, y, s, dataidx_map = split_dataset(data, users_group_dict, setting, num_parties)

        for client in range(num_parties):
            # Data to be written
            dictionary = {
                "person": dataidx_map[client],
                "image": X[client],
                "target": y[client],
                "sens": s[client],
            }
            
            with open(f"/home/tongnian/leaf/data/celeba/data/non-iid/client_{client}.json", "w") as outfile:
                json.dump(dictionary, outfile)




if __name__ == '__main__':
	main()