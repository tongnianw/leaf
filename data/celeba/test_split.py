import numpy as np
from collections import defaultdict
import json
import random
np.random.seed(0)

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    Split data labels into n clients based on Dirichlet distribution

    '''
    n_classes = int(train_labels.max()+1)
    print(n_clients, n_classes)
    # generate a Dirichlet distribution of client weights for each class. 
    # It ensures that the distribution of labels is non-IID across clients
    # np.random.dirichlet function takes n_clients as the number of output samples 
    # and n_classes as the number of categories. 
    # The concentration parameter alpha is used to control the distribution's shape
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # This list comprehension creates a list of arrays. 
    # Each array contains the indices of data points for a specific class.
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # store the indices of data points assigned to each client
    client_idcs = [[] for _ in range(n_clients)]
    # distribute the data points to clients based on the Dirichlet-distributed weights
    # It iterates through each class and assigns data points to clients according to 
    # the weights determined by the Dirichlet distribution
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split: split the class indices into segments based on the cumulative weights
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))
                                    ):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    print([len(client_idcs[i]) for i in range(len(client_idcs))])

    return client_idcs

def add_arrays(array1, array2):

  return [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(array1, array2)]


def main():

    # 1. Open the JSON file for reading
    with open('/home/tongnian/leaf/data/celeba/data/all_data/all_data.json', 'r') as file:
        # 2. Parse the JSON data using json.load()
        data = json.load(file)

    num_samples = data['num_samples']
    users = data['users']
    user_data = data['user_data']

    # s = data['user_data'][users[:]]['s']
    # print(s)

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

    n_clients = 5
    alpha = 1.0
    client_idcs = dirichlet_split_noniid(np.array(users_group_dict['s']), alpha, n_clients)




if __name__ == '__main__':
	main()