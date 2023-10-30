import json
import numpy as np
import os
import random


TARGET_NAME = 'Young'
GROUP_NAME = 'Male'
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_metadata():
	f_identities = open(os.path.join(
		parent_path, 'data', 'raw', 'identity_CelebA.txt'), 'r')
	identities = f_identities.read().split('\n')
	print(len(identities))

	f_attributes = open(os.path.join(
		parent_path, 'data', 'raw', 'list_attr_celeba.txt'), 'r')
	attributes = f_attributes.read().split('\n')
	print(len(attributes))
	print(attributes[1])

	sample_identities = identities
	# # Randomly select 10,000 rows from list_attr_celeba
	# sample_identities = random.sample(identities, 10000)
	print(len(sample_identities))

	return sample_identities, attributes


def get_celebrities_and_images(identities):
	all_celebs = {}

	for line in identities:
		info = line.split()
		if len(info) < 2:
			continue
		image, celeb = info[0], info[1]
		if celeb not in all_celebs:
			all_celebs[celeb] = []
		all_celebs[celeb].append(image)

	# print(all_celebs['5335']) # ['198787.jpg', '199613.jpg', '199972.jpg']
	good_celebs = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) >= 0}

	return good_celebs


def _get_celebrities_by_image(identities):
	good_images = {}
	for c in identities:
		images = identities[c]
		for img in images:
			good_images[img] = c
	return good_images


def get_celebrities_and_target(celebrities, attributes, attribute_name=TARGET_NAME, group_name=GROUP_NAME):
	col_names = attributes[1]
	col_idx = col_names.split().index(attribute_name)
	group_col_idx = col_names.split().index(group_name)

	celeb_attributes = {}
	group_attributes = {}
	
	good_images = _get_celebrities_by_image(celebrities)
	# print(good_images) # {...,'198787.jpg': '5335', '199613.jpg': '5335', '199972.jpg': '5335'}
	print(len(good_images))

	for line in attributes[2:]:
		info = line.split()
		if len(info) == 0:
			continue

		image = info[0]
		if image not in good_images:
			continue
		
		celeb = good_images[image]
		att = (int(info[1:][col_idx]) + 1) / 2 # set target 1 & -1 --> 1 and 0
		group_att = (int(info[1:][group_col_idx]) + 1) / 2 # set target 1 & -1 --> 1 and 0
		
		if celeb not in celeb_attributes:
			celeb_attributes[celeb] = []
		celeb_attributes[celeb].append(att)

		if celeb not in group_attributes:
			group_attributes[celeb] = []
		group_attributes[celeb].append(group_att)

	# print(celeb_attributes['5335']) # target: [0.0, 0.0]
	# print(group_attributes['5335']) # group: [1.0, 0.0]
	return celeb_attributes, group_attributes


def build_json_format(celebrities, targets, group):
	all_data = {}

	celeb_keys = [c for c in celebrities]
	print(len(celeb_keys))
	num_samples = [len(celebrities[c]) for c in celeb_keys]
	data = {c: {'x': celebrities[c], 'y': targets[c], 's': group[c]} for c in celebrities}

	all_data['users'] = celeb_keys
	all_data['num_samples'] = num_samples
	all_data['user_data'] = data
	return all_data


def write_json(json_data):
	file_name = 'all_data.json'
	dir_path = os.path.join(parent_path, 'data', 'all_data')

	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

	file_path = os.path.join(dir_path, file_name)

	print('writing {}'.format(file_name))
	with open(file_path, 'w') as outfile:
		json.dump(json_data, outfile)


def main():
	identities, attributes = get_metadata()
	celebrities = get_celebrities_and_images(identities)
	targets, group = get_celebrities_and_target(celebrities, attributes)

	json_data = build_json_format(celebrities, targets, group)
	write_json(json_data)
	# 202599 images, 10177 unique celebs

if __name__ == '__main__':
	main()


