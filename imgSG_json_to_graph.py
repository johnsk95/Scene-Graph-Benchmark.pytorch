import json
import numpy as np

class Node:
    def __init__(self, name, attr, node_id):
        self.name = name
        self.attr = attr
        self.node_id = node_id
        self.parent = []
        self.child = []
    
    def get_name(self):
        return self.name
    
    def get_attr(self):
        if self.attr:
            return self.attr
        else:
            return ""

    def get_node_id(self):
        return self.node_id
    
    def get_parent(self):
        return self.parent

    def get_child(self):
        return self.child

    def update_child(self, node):
        self.child.append(node)

    def update_parent(self, node):
        self.parent.append(node)
        
class Edge:
    def __init__(self, sub, rel, obj):
        self.sub = sub
        self.rel = rel
        self.obj = obj

    def get_edge(self):
        return (self.sub, self.rel, self.obj)

    def get_triplet(self):
        return (str(self.sub.get_attr() + " " + self.sub.name + " " \
            + self.rel + " " + self.obj.get_attr() + " " + self.obj.name))

def find_leaves(nodes):
    """find all leaf nodes"""
    leaves = []
    for node in nodes:
        if not node.get_child():
            leaves.append(node)
    return leaves

def find_node(node_id, nodes):
    """find node based on the node_id"""
    for node in nodes:
        if node.node_id == node_id:
            return node
    return None

def find_edges_with_leaves(edges):
    leaves_edge = []
    for edge in edges:
        if not edge.obj.get_child():
            leaves_edge.append(edge)
    return leaves_edge

def create_graph(pred, data):
    # custom_prediction = json.load(open('custom_prediction.json'))
    # custom_data_info = json.load(open('custom_data_info.json'))
    custom_prediction = json.load(open(pred))
    custom_data_info = json.load(open(data))

    # parameters
    image_idx = 5
    box_topk = 10 # select top k bounding boxes
    rel_topk = 40 # select top k relationships
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']
    ind_to_attributes = custom_data_info['ind_to_attributes']

    image_path = custom_data_info['idx_to_files'][image_idx]
    boxes = custom_prediction[str(image_idx)]['bbox'][:box_topk]
    box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
    # box_scores = custom_prediction[str(image_idx)]['bbox_scores'][:box_topk]
    box_attrs = custom_prediction[str(image_idx)]['bbox_attrs'][:box_topk]
    all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
    # all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
    all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

    # box labels
    for i in range(len(box_labels)):
        box_labels[i] = ind_to_classes[box_labels[i]]
    #print(box_labels)
    
    # Attributes for the boxes
    for i in range(len(box_attrs)):
        attr = ind_to_attributes[np.argmax(box_attrs[i])]
        if attr == 'on':
            idx = np.argsort(box_attrs[i])
            box_attrs[i] = ind_to_attributes[idx[-2]]
        else:
            # box_attrs[i] = ind_to_attributes[np.argmax(box_attrs[i])]
            box_attrs[i] = ind_to_attributes[box_attrs[i]]
    #print(box_attrs)
    
    # Relations between the boxes and initializing nodes
    rel_labels = []
    # rel_scores = []
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            #rel_scores.append(all_rel_scores[i])
            label = str(all_rel_pairs[i][0]) + '_' + box_attrs[all_rel_pairs[i][0]] + ' ' + box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[all_rel_labels[i]] + ' => ' + str(all_rel_pairs[i][1]) + '_' + box_attrs[all_rel_pairs[i][1]] + ' ' + box_labels[all_rel_pairs[i][1]]
            rel_labels.append(label)
    rel_labels = rel_labels[:rel_topk]
    # rel_scores = rel_scores[:rel_topk]
    #print(rel_labels)

    # initialize nodes
    nodes = []
    for node_id, name in enumerate(box_labels):
        nodes.append(Node(name, box_attrs[node_id], node_id))
    #print(len(nodes))
    
    # initialize edges
    edges = []
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            # find the node using these information?
            sub_id = all_rel_pairs[i][0]
            sub_node = find_node(sub_id,nodes)
            rel = ind_to_predicates[all_rel_labels[i]]
            obj_id = all_rel_pairs[i][1]
            obj_node = find_node(obj_id,nodes)
            sub_node.update_child(obj_node)
            obj_node.update_parent(sub_node)
            edges.append(Edge(sub_node, rel, obj_node))
    edges = edges[:rel_topk]

    return nodes, edges, boxes, image_path
