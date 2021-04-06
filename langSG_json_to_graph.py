# json to graph object
import json

class Node:
    def __init__(self, name, attr):
        self.name = name
        self.attr = attr
        self.parent = []
        self.child = []
    
    def get_name(self):
        return self.name
    
    def get_attr(self):
        return self.attr
    
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

def find_leaves(nodes):
    """find all leaf nodes"""
    leaves = []
    for node in nodes:
        if not node.get_child():
            leaves.append(node)
    return leaves

def find_edges_with_leaves(edges):
    leaves_edge = []
    for edge in edges:
        if not edge.obj.get_child():
            leaves_edge.append(edge)
    return leaves_edge

def find_edges_with_root(edges):
    root_edge = []
    for edge in edges:
        if not edge.sub.get_parent():
            root_edge.append(edge)
    return root_edge

def create_graph(filename):
    with open(filename) as f:
        data = json.load(f)
    
    nodes = []
    edges = []

    objs = data['objects'] # list of objects
    for i, obj in enumerate(objs):
        name = obj['names'][0]
        attribute = None
        if data['attributes']:
            for attr in data['attributes']:
                if attr['subject'] == i:
                    attribute = attr['attribute']
        nodes.append(Node(name, attribute))

    rels = data['relationships'] # list of relationships
    for rel in rels:
        sub_idx = rel['subject'] 
        obj_idx = rel['object']
        rel = rel['predicate']
        sub = nodes[sub_idx]
        obj = nodes[obj_idx]
        edges.append(Edge(sub, rel, obj))

    # update child and parent for each node
    for edge in edges:
        sub_node = edge.sub
        obj_node = edge.obj
        sub_node.update_child(obj_node)
        obj_node.update_parent(sub_node)

    return nodes, edges