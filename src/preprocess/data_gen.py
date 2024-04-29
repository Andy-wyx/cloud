import json
import copy
import random

def get_id_ood_category_set(data):
    #Define ID and OOD categories

    #last_id of each supercategory as ood category id set
    # supercat_set = list(set([cat['supercategory'] for cat in data['categories']]))

    # ood_category_set = []
    # for supercat in supercat_set:
    #     cats = [cat['id'] for cat in data['categories'] if cat['supercategory']==supercat]
    #     ood_category_set.append(max(cats))
    # ood_category_set = sorted(ood_category_set)
    # ood_category_set = ood_category_set[1:] #keep persons as id
    # #other ids as id set
    total_ids = [cat['id'] for cat in data['categories']]
    ood_category_set = total_ids[69:]
    id_category_set = list(set([cat['id'] for cat in data['categories']]).difference(set(ood_category_set)))

    return id_category_set,ood_category_set

def split_data_by_category(data, category_set):
    data_copy = copy.deepcopy(data)
    # Filter categories
    filtered_categories = [cat for cat in data_copy['categories'] if cat['id'] in category_set]
    valid_ids = {cat['id'] for cat in filtered_categories}

    # Filter annotations based on valid category_ids
    filtered_annotations = [anno for anno in data_copy['annotations'] if anno['category_id'] in valid_ids]
    valid_image_ids = {anno['image_id'] for anno in filtered_annotations}

    # Filter images based on remaining valid image_ids
    filtered_images = [img for img in data_copy['images'] if img['id'] in valid_image_ids]

    # Update the data dictionary with filtered data
    data_copy['categories'] = filtered_categories
    data_copy['annotations'] = filtered_annotations
    data_copy['images'] = filtered_images

    return data_copy

def filter_mixed(id_data, ood_data):
    id_data_copy = copy.deepcopy(id_data)
    ood_data_copy = copy.deepcopy(ood_data)
    id_image_ids = set([img['id'] for img in id_data_copy['images']])
    ood_image_ids = set([img['id'] for img in ood_data_copy['images']])
    mixed_image_ids = list(set(id_image_ids).intersection(set(ood_image_ids)))
    filtered_id_image_ids = list(id_image_ids.difference(set(mixed_image_ids)))
    filtered_ood_image_ids = list(ood_image_ids.difference(set(mixed_image_ids)))
    filtered_id_images = [img for img in id_data_copy['images'] if img['id'] not in mixed_image_ids]
    filtered_ood_images = [img for img in ood_data_copy['images'] if img['id'] not in mixed_image_ids]
    filtered_id_annos = [anno for anno in id_data_copy['annotations'] if anno['image_id'] in filtered_id_image_ids]
    filtered_ood_annos = [anno for anno in ood_data_copy['annotations'] if anno['image_id'] in filtered_ood_image_ids]

    id_category_ids = []
    ood_category_ids = []
    for i in filtered_id_annos:
        id_category_ids.append(i['category_id'])
    for i in filtered_ood_annos:
        ood_category_ids.append(i['category_id'])
    filtered_id_categories = [cat for cat in id_data_copy['categories'] if cat['id'] in id_category_ids]
    filtered_ood_categories = [cat for cat in ood_data_copy['categories'] if cat['id'] in ood_category_ids]

    id_data_copy['categories'] = filtered_id_categories
    id_data_copy['annotations'] = filtered_id_annos
    id_data_copy['images'] = filtered_id_images

    ood_data_copy['categories'] = filtered_ood_categories
    ood_data_copy['annotations'] = filtered_ood_annos
    ood_data_copy['images'] = filtered_ood_images

    return id_data_copy,ood_data_copy

def gen_id_ood_dataset(id_category_set,ood_category_set,filepath,mapping_dict):
    def save_json(data, new_filename):
        with open(new_filename, 'w') as file:
            json.dump(data, file)
            
    with open(filepath, 'r') as file:
        data = json.load(file)

    id_data = split_data_by_category(data, id_category_set)
    ood_data = split_data_by_category(data, ood_category_set)
    filtered_id_data, filtered_ood_data = filter_mixed(id_data=id_data, ood_data=ood_data)
    #do the mapping
    #for data['categories']
    for cat in filtered_id_data['categories']:
        cat['id'] = mapping_dict[cat['id']]
    for cat in filtered_ood_data['categories']:
        cat['id'] = mapping_dict[cat['id']]
    #for data['annotations']
    for anno in filtered_id_data['annotations']:
        anno['category_id'] = mapping_dict[anno['category_id']]
    for anno in filtered_ood_data['annotations']:
        anno['category_id'] = mapping_dict[anno['category_id']]

    save_json(filtered_id_data, data_folder_path+'id_pretrain.json')
    print('id pretrain dataset saved')
    save_json(filtered_ood_data, data_folder_path+'ood_test.json')
    print('ood test dataset saved')

def gen_id_testing_dataset(id_data_file_path,percentage=0.3):
    with open(id_data_file_path, 'r') as file:
        data = json.load(file)
    
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    # Calculate the split index
    split_index = int(len(image_ids) * percentage)
    
    # Split image IDs into training and testing sets # TODO
    test_image_ids = set(image_ids[:split_index])

    test_images = [img for img in data['images'] if img['id'] in test_image_ids]
    test_annos = [anno for anno in data['annotations'] if anno['image_id'] in test_image_ids]

    data['images']=test_images
    data['annotations'] = test_annos

    def save_json(data, new_filename):
        with open(new_filename, 'w') as file:
            json.dump(data, file)
    save_json(data, data_folder_path+'id_test.json')
    print('id test dataset saved')

def get_json_info(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(file_path.split('/')[-1]+":")
    print("\ttop-level properties:", list(data.keys()))
    print("\tnumber of images:", f"{len(data['images'])}/{num_of_images_all}")
    print("\tnumber of annotations:", f"{len(data['annotations'])}/{num_of_annotations_all}")
    category_set = [cat['id'] for cat in data['categories']]
    print("\tnumber of categories:", f"{len(data['categories'])}/{num_of_categories_all}",'including category ids:', list(category_set))

if __name__ == '__main__':
    #specify your relative path for data folder
    data_folder_path = "../../data/coco/annotations/"
    filepath = data_folder_path+"instances_val2017.json"
    generated_pretrain_filepath = data_folder_path+"id_test.json"

    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # we need to use this mapping dict to transform all category id in data[categories] and data[annotations] back into a perfectly continous list from 1 to 80
    mapping = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
        11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20,
        22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
        35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
        46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
        56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60,
        67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70,
        80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
    }

    num_of_images_all,num_of_categories_all,num_of_annotations_all = len(data['images']),len(data['categories']),len(data['annotations'])

    id_category_set,ood_category_set = get_id_ood_category_set(data)

    #generate id and ood dataset
    gen_id_ood_dataset(id_category_set,ood_category_set,filepath=filepath,mapping_dict=mapping)

    # Generate ID testing dataset:Use part of the id data as id_test data for model inference
    gen_id_testing_dataset(data_folder_path+"id_pretrain.json",percentage=0.2)

    # display information of the generated json file
    get_json_info(data_folder_path+'id_pretrain.json')
    get_json_info(data_folder_path+'id_test.json')
    get_json_info(data_folder_path+'ood_test.json')

    #display id class names for use in the model.
    
    with open(generated_pretrain_filepath, 'r') as file:
        data = json.load(file)
    ids = [cat['id'] for cat in data['categories']]
    names = [cat['name'] for cat in data['categories']]
    print('class names of in distribution testing dataset:',names)