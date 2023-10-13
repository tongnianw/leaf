import numpy as np
import json
import random

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

        # Identify the minority subgroup
        minority_group = {"y": 0.0, "s": 0.0}
        # Separate users into majority and minority groups
        majority_users = []
        minority_users = []
        
        # filtering the users into two groups: the "majority group" and the "minority group" 
        # based on the values of "y" (label) and "s" (group or sensitive attribute) in the dataset.
        for user_id, data in user_data.items():
            # (1) checks if the label ("y") of a user does not match the label of the "minority group"
            # (2) checks if the group or sensitive attribute ("s") of a user does not match the group of the "minority group"
            if data["y"][0] != minority_group["y"] or data["s"][0] != minority_group["s"]:
                majority_users.append(user_id)
            else:
                minority_users.append(user_id)
        
        # Shuffle the user lists
        random.shuffle(majority_users)
        random.shuffle(minority_users)
        
        # Determine the number of parties for each ratio
        num_majority_parties = num_parties // 2
        num_minority_parties = num_parties - num_majority_parties
        
        # Create party assignments
        party_assignments = []

        # Assign users to parties while maintaining the desired ratio
        for i in range(num_majority_parties):
            party_size = user_counts[i]
            print('party_size:', party_size)
            party_assignments.extend(majority_users[:party_size])
            majority_users = majority_users[party_size:]
        
        for i in range(num_minority_parties):
            party_size = user_counts[i + num_majority_parties]
            print('party_size:', party_size)
            party_assignments.extend(minority_users[:party_size])
            minority_users = minority_users[party_size:]
    
        # Create party-specific datasets
        party_datasets = []
        
        for i in range(num_parties):
            party_data = {
                "users": party_assignments[i],
                "num_samples": [dataset["num_samples"][users.index(user)] for user in party_assignments[i]],
                "user_data": {user: user_data[user] for user in party_assignments[i]}
            }
            party_datasets.append(party_data)
        
        return



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
    setting = 'i'
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