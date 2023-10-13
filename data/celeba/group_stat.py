import os




def main():

    f_attributes = open('/home/tongnian/leaf/data/celeba/data/raw/list_attr_celeba.txt', 'r')
    attributes = f_attributes.read().split('\n')

    col_names = attributes[1]
    print(col_names)

    col_idx = col_names.split().index('Male')
    print(col_idx)

    # Initialize a dictionary to store the counts of unique values in the 'sex' column
    sex_counts = {}
    
    # Iterate through the data (skipping the header)
    for line in attributes[2:]:
        info = line.split()

        if len(info) == 0:
            print('len = 0!')
            continue
        # Get the 'sex' value for the current row
        group_value = info[col_idx]

        # Update the count in the dictionary
        if group_value in sex_counts:
            sex_counts[group_value] += 1
        else:
            sex_counts[group_value] = 1

    print(sex_counts) # {'1': 92189, '-1': 110410}

if __name__ == '__main__':
	main()