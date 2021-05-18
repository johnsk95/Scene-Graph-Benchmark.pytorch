import json


custom_data_info = json.load(open('custom_data_info.json'))

ind_to_classes = custom_data_info['ind_to_classes']
ind_to_predicates = custom_data_info['ind_to_predicates']
ind_to_attributes = custom_data_info['ind_to_attributes']

labels = ['table', 'apple', 'apple', 'apple','box', 'book', 'book']
comparing_list = ind_to_classes

indices = []

for label in labels:
    indices.append(comparing_list.index(label))

print(indices)