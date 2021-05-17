import imgSG_json_to_graph as ISG
import langSG_json_to_graph as LSG
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw

# word embedding model
model = SentenceTransformer('stsb-roberta-base')

# make SG outputs into graphs
lang_nodes, lang_edges = LSG.create_graph('lang_sg_result.json')
img_nodes, img_edges, boxes, image_path = ISG.create_graph('prediction.json', 'custom_data_info_2.json')
# get leaf nodes from lsg
leaf_edges = LSG.find_edges_with_leaves(lang_edges)
# candidate list
cand_list = []
# asking list
ask_list = []

def isempty(a):
    if a:
        return False
    else: 
        return True

def compare_rels(lang_edge, img_edge):
    lang_rel_emb = model.encode(lang_edge.rel, convert_to_tensor=True)
    img_rel_emb = model.encode(img_edge.rel, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(lang_rel_emb, img_rel_emb)
    return cosine_scores

def compare_attr(lang_node, img_node):
    lang_attr_emb = model.encode(lang_node.attr, convert_to_tensor=True)
    img_attr_emb = model.encode(img_node.attr, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(lang_attr_emb, img_attr_emb)
    return cosine_scores

def compare_name(lang_node, img_node):
    lang_name_emb = model.encode(lang_node.name, convert_to_tensor=True)
    img_name_emb = model.encode(img_node.name, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(lang_name_emb, img_name_emb)
    return cosine_scores

def ground_or_leaf(leaf_edge, img_edge):
    global leaf_edges
    global cand_list
    global lang_edges
    # if root edge
    if isempty(leaf_edge.sub.parent):
        # add to cand list
        cand_list.append(img_edge)
        print(img_edge.get_triplet())
    # edge has parent edge
    else:
        # find curr->parent edge
        parent_edge = find_edge(leaf_edge.sub, lang_edges)
        leaf_edges.append(parent_edge)

def find_edge(currNode, edges):
    for edge in edges:
        if edge.sub == currNode.parent and edge.obj == currNode:
            return edge
    return None

def find_unique(input_list):
    """If subject is grounded find unique relation for that subject to present to the user.
    If subject is not grounded and we have subject candidates, find unique relation for 
    each of them and ask them back to the user.
    regardless, this function should find unique relation for the subject node/s"""
    # ex) {'white plate on wooden table': [7,5,4,3,8]}
    # ex) {'white plate on wooden table across the chair'}
    # cand_list = [[7,5,4,3,8], [4,3,8], [3,8]] => [3,8] => find unique => ask
    # if multiple components in reduced cand list --> find unique edges for each subject node
    # merge first and if nothing exists make set and find unique
    # subject not exist in image --> no matching subjects in cand lists found
    # then we know the subject nodes that appears in 'white plate on wooden table'
    # given these candidates:
    # 1. find edges that contain [7,5,4,3,8] as subjects
    # 2. 
    # create dictionary that contains list of edges that contains subNode as subject
    # input = [7,5,4,3,8]
    global img_edges
    # for each candidate find all edges that contains candidate as subject node
    # extract subject id from candidate list
    if isempty(input_list):
        return input_list
    
    cand_ids = []
    for edge in input_list:
        cand_ids.append(edge.sub.get_node_id())
    print("cand_id:", cand_ids)
    
    edges_contain_cand = []
    for cand in cand_ids:
        temp_edges = []
        for edge in img_edges:
            if cand == edge.sub.get_node_id():
                temp_edges.append(edge)
        edges_contain_cand.append(temp_edges)
    print("edges_contain_cand: ", edges_contain_cand)
    # at this point [[edges that contain 7],[edges that contain 5],...]
    # to count occurences of the edges, we need to change the edge -> string
    # ex) 7(plate) in front of 9(table) -> plate in front of table
    str_edges_contain_cand = []
    for edge_list in edges_contain_cand:
        temp_str_edges = []
        for edge in edge_list:
            str_edge = edge.get_triplet()
            temp_str_edges.append(str_edge)
        str_edges_contain_cand.append(temp_str_edges)
    print("str_edges_contain_cand: ", str_edges_contain_cand)
    # count occurrences of the edges
    # convert str_edges_contain_cand to list of dictionaries
    # ex) [{white plate in front of table: 3, ...}, {}, {}]
    flat_list = [item for sublist in str_edges_contain_cand for item in sublist]
    cnt_str_edges_contain_cand = []
    for edge_list in edges_contain_cand:
        freq = {}
        for edge in edge_list:
            freq[edge] = flat_list.count(edge.get_triplet())
        cnt_str_edges_contain_cand.append(freq)
    print("cnt_str_edges_contain_cand: ", cnt_str_edges_contain_cand)
    # now we have this form [{white plate in front of table: 3, ...}, {}, {}]
    # for each dict, pick key with smallest value for each list
    # min(dict, key=d.get)
    return_list = []
    for dict in cnt_str_edges_contain_cand:
        min_occurred = min(dict.keys(), key=(lambda k: dict[k]))
        return_list.append(min_occurred)
    print("return_list: ", return_list)
    return return_list

def ask_questions(curr_leaf_idx):
    global leaf_edges
    global lang_edges
    global ask_list
    global cand_list
    print('ask questions')
    if len(ask_list) == 0:
        print('ask list empty: no grounding')
    elif len(ask_list) == 1:
        print(ask_list[0].get_triplet())
        feedback = input("is this right?(y/n) ")
        if feedback == 'y':
            #cand_list.append(ask_list[0])
            cand_list.append(ask_list[0])
        #cand_list = find_unique(cand_list)
    else:
        # print(ask_list)
        # samedic = defaultdict(list)
        # strings = []
        # for img_edge in ask_list:
        #     strings.append(img_edge.get_triplet())
        # print(strings)
        # print(find_unique(ask_list))
        # # construct dict of duplicate triplets
        # # {triplet: index of edge in ask_list}
        # for i, triplet in enumerate(strings):
        #     # samedic[triplet].append(ask_list[i].sub.get_node_id())
        #     samedic[triplet].append(i)
        # feedback_yn = None
        # for items in samedic:
        #     # check for same triplets
        #     if len(samedic[items]) > 1:
        #         # draw bounding box on each sub node
        #         for idx in samedic[items]:
        #             display_bbox(ask_list[idx].sub)
        #             feedback_yn = input(f'Is it this item? ')
        #             if feedback_yn == 'y':
        #                 #cand_list.append(ask_list[idx])
        #                 cand_list.append(find_unique(list(ask_list[idx])))
        #                 break
        #         if feedback_yn == 'y':
        #             break
        # if feedback_yn != 'y':
        #     for idx, ask in enumerate(ask_list):
        #         # list options to human
        #         print(idx, ask.get_triplet())
        #         # accept human input
        #     feedback_idx = input("which one? ")
        #     selected_edge = ask_list[int(feedback_idx)]
        #     # if selected edge is the root edge
        #     if isempty(leaf_edges[curr_leaf_idx].sub.parent):
        #         # insert the edge in the candidate list
        #         cand_list.append(find_unique(list(selected_edge)))
        #     # if edge has parent edge
        #     else:
        #         parent_edge = find_edge(leaf_edges[curr_leaf_idx].sub, lang_edges)
        #         # next edge to be looked into should be the parent
        #         leaf_edges.insert(curr_leaf_idx+1, parent_edge)
        unique_list = find_unique(ask_list)
        print(unique_list)
        for idx, ask in enumerate(unique_list):
            # list options
            print(f'[{idx}] {ask.get_triplet()}')
        feedback_idx = input("which one? ")
        selected = ask_list[int(feedback_idx)]
        cand_list.append(selected)
    # empty ask_list
    ask_list = []

def ask_with_image(edges):
    global cand_list
    samedic = defaultdict(list)
    strings = []
    for img_edge in edges:
        strings.append(img_edge.get_triplet())
    # construct dict of duplicate triplets
    for i, triplet in enumerate(strings):
        samedic[triplet].append(i)

    for items in samedic:
        if len(samedic[items]) > 1:
            # draw bounding box on each sub node
            feedback_yn = None
            for idx in samedic[items]:
                display_bbox(edges[idx].sub)
                feedback_yn = input('Is it this item? ')
                if feedback_yn == 'y':
                    cand_list.append(edges[idx])
                    break
            if feedback_yn == 'y':   
                break
        else:
            ask_list.append(edges[samedic[items][0]])

def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)

def display_bbox(node):
    size = get_size(Image.open(image_path).size)
    pic = Image.open(image_path).resize(size)
    draw = ImageDraw.Draw(pic)
    bbox = boxes[node.get_node_id()]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline='red')
    pic.show()

# when no edge is formed in LSG
if isempty(leaf_edges):
    print('no edge formed')
    # ex. give me the cup (only contains subject)
    # find all edges containing root node in ISG
    vref = lang_nodes[0]
    img_root_edges = []
    for img_edge in img_edges:
        if compare_name(vref, img_edge.sub) > 0.8:
            img_root_edges.append(img_edge)
    # match attribute
    for img_root_edge in img_root_edges.copy():
        if vref.attr is not None:
            # remove img edges with subject attr not matching
            if compare_attr(vref, img_root_edge.sub) < 0.8:
                img_root_edges.remove(img_root_edge)
    #print(img_root_edges)
    if not isempty(img_root_edges):
        filtered_img_edges = []
        # compare node id of candidate edges
        checked_ids = []
        for img_edge in img_root_edges:
            sub_id = img_edge.sub.get_node_id()
            # add only one edge from each object id
            if sub_id not in checked_ids:
                checked_ids.append(sub_id)
                # cand_list.append(img_edge)
                filtered_img_edges.append(img_edge)
        # cand_list = img_root_edges
        ####################################################
        # compare triplet strings
        # create dict containing indices of same items
        samedic = defaultdict(list)
        strings = []
        for img_edge in filtered_img_edges:
            strings.append(img_edge.get_triplet())
        for i, triplet in enumerate(strings):
            samedic[triplet].append(i)
        # ['white plate on wooden table', 'white plate on wooden table', 'white plate near lamp', 'white plate near black cat', 'white plate on wooden table', 'cat']
        # -> {white plate on wooden table: [0,1,2], white plate near lamp: [3,4]}
        # goal: extract unique triplet per subject
        for items in samedic:
            if len(samedic[items]) > 1:
                # draw bounding box on each sub node
                feedback_yn = None
                for idx in samedic[items]:
                    display_bbox(filtered_img_edges[idx].sub)
                    feedback_yn = input('Is it this item? ')
                    if feedback_yn == 'y':
                        cand_list.append(filtered_img_edges[idx])
                        break
                if feedback_yn == 'y':   
                    break
            else:
                cand_list.append(filtered_img_edges[samedic[items][0]])
                # test

        #####################################################
    else:
        print('no ', vref.get_name())
    #print(cand_list)
else:
    # "black cat on wooden table"
    # "white plate on wooden table"
    for i, leaf_edge in enumerate(leaf_edges):
        # temp
        # temp_cand_list = []
        # Object matching (lang -> image)
        obj_matching_edges = []
        for img_edge in img_edges:
            # compare nodes using word embedding
            if compare_name(leaf_edge.obj, img_edge.obj) > 0.8:
                obj_matching_edges.append(img_edge)
        # One edge with same object match
        if len(obj_matching_edges) == 1:
            print('one obj match')
            # Check if sub also matches
            img_edge = obj_matching_edges[0]
            if compare_name(leaf_edge.sub, img_edge.sub) > 0.8:
                # One pred match
                if compare_rels(leaf_edge, img_edge) > 0.8:
                    # has parent -> add to leaf, root -> add to grounding cand
                    ground_or_leaf(leaf_edge, img_edge)
                # No pred match
                else:
                    # add to ask list and ask
                    ask_list.append(img_edge)
                    ask_questions(i)
            # No subject matches
            else:
                # add to ask list and ask
                ask_list.append(img_edge)
                ask_questions(i)
        # multiple edges with object match
        elif len(obj_matching_edges) > 1:
            print('mult obj match')
            sub_match = []
            # look at the subject of the edges
            for img_edge in obj_matching_edges:
                if compare_name(leaf_edge.sub, img_edge.sub) > 0.8:
                    # collect edges that match sub
                    sub_match.append(img_edge)
            # One match sub
            if len(sub_match) == 1:
                print('mult obj, one sub match')
                # matching pred
                if compare_rels(sub_match[0], leaf_edge) > 0.8:
                    ground_or_leaf(leaf_edge, sub_match[0])
                # if no matching pred
                else:
                    print('mult obj, one sub match, no pred match')
                    ask_list.append(sub_match[0])
                    ask_questions(i)
            # multiple subject matches
            elif len(sub_match) > 1:
                print('mult obj, mult sub match')
                sub_pred_match = []
                # match pred of each edge
                for sub_edge in sub_match:
                    if compare_rels(leaf_edge, sub_edge) > 0.8:
                        sub_pred_match.append(sub_edge)
                # one pred match
                if len(sub_pred_match) == 1:
                    print('mult obj, mult sub match, one pred match')
                    ground_or_leaf(leaf_edge, sub_pred_match[0])
                # multiple pred match
                elif len(sub_pred_match) > 1:
                    print('mult obj, mult sub match, mult pred match')
                    attr_match = sub_pred_match.copy()
                    # match subject attribute
                    for sub_pred_edge in sub_pred_match:
                        if leaf_edge.obj.attr is not None:
                            # print(leaf_edge.obj.get_attr())
                            # compare object attributes
                            if compare_attr(leaf_edge.obj, sub_pred_edge.obj) < 0.8:
                                attr_match.remove(sub_pred_edge)
                    # One matching attribute
                    if len(attr_match) == 1:
                        print('mult obj, mult sub, mult pred, one attr match')
                        ask_list.append(attr_match[0])
                        ask_questions(i)
                    # multiple matching attribute
                    elif len(attr_match) > 1:
                        print('mult obj, mult sub, mult pred, mult attr match')
                        ask_list.extend(attr_match)
                        ask_questions(i)
                    # no matching attribute
                    else:
                        print('mult obj, mult sub, mult pred, no attr match')
                        # ask all in spo match list (this, this, ...)
                        ask_list.extend(sub_pred_match)
                        ask_questions(i)
                # no pred match
                else:
                    print('mult obj, mult sub match, no pred match')
                    # choose most similar predicate and ask
                    sim_scores = []
                    for sub_edge in sub_match:
                        sim_score = compare_rels(leaf_edge, sub_edge)
                        if leaf_edge.sub.attr is not None:
                            sim_score += compare_attr(leaf_edge.sub, sub_edge.sub)
                        sim_scores.append(sim_score)
                    max_score_idx = np.argmax(sim_scores)
                    ask_list.append(sub_match[max_score_idx])
                    ask_questions(i)
            # no matching sub
            else:
                print('mult obj, no sub match')
                pred_obj_match = []
                sim_scores = []
                for obj_matching_edge in obj_matching_edges:
                    sim_score = compare_rels(obj_matching_edge, leaf_edge)
                    if sim_score > 0.8:
                        pred_obj_match.append(obj_matching_edge)
                    sim_scores.append(sim_score)
                if len(pred_obj_match) == 1:
                    print('mult obj, no sub match, one pred match')
                    ask_list.append(pred_obj_match[0])
                    ask_questions(i)
                # else:
                elif len(pred_obj_match) > 1:
                    print('mult obj, no sub match, mult or no pred match')
                    # if there are multiple objects with similar pred
                    # return one with most similar sub
                    sim_sub = []
                    sub_sim_scores = []
                    for cand_sub in pred_obj_match:
                        sim_score = compare_name(leaf_edge.sub, cand_sub.sub)
                        sub_sim_scores.append(sim_score)
                        sim_sub.append(cand_sub.sub.name)
                    print('pred_obj_match: ', pred_obj_match)
                    sub_max_score_idx = np.argmax(sub_sim_scores)
                    ask_list.append(pred_obj_match[sub_max_score_idx])
                    # ask_list.extend(pred_obj_match)
                    ask_questions(i)
                else:
                    ask_list.extend(obj_matching_edges)
                    ask_questions(i)
        # no matching edge with object match
        else:
            print('no obj match')
            if i < len(leaf_edges)-1:
                # pass if additional leaf nodes exist
                pass
            else:
                # get root node from LSG
                lang_root_edges = LSG.find_edges_with_root(lang_edges)
                # match subject
                # find all edges containing root node in ISG
                img_root_edges = []
                vref = lang_root_edges[0].sub
                for img_edge in img_edges:
                    if compare_name(vref, img_edge.sub) > 0.8:
                        # edges in ISG that contain vref node (subject)
                        img_root_edges.append(img_edge)
                if len(img_root_edges) > 0:
                    print('no obj, one or more sub matches')
                    # match attributes
                    for lang_root_edge in lang_root_edges:
                            for img_root_edge in img_root_edges.copy():
                                if lang_root_edge.sub.attr is not None:
                                    # remove edges with mismatching attributes
                                    if compare_attr(lang_root_edge.sub, img_root_edge.sub) < 0.8:
                                        img_root_edges.remove(img_root_edge)
                                # compare predicates
                                # remove edges with mismatching predicates
                                if compare_rels(lang_root_edge, img_root_edge) < 0.8:
                                    img_root_edges.remove(img_root_edge)
                    # ask filtered edges
                    ask_list.extend(img_root_edges)
                    ask_questions(i)
                else:
                    print('no obj, no sub match')

if not isempty(cand_list):
    if len(cand_list) == 1:
        # achieved grounding
        print('grounding achieved!')
        print(cand_list[0].get_triplet())
        display_bbox(cand_list[0].sub)
    else:
        print('ask candidates')
        # multiple candidates
        # ask unique relation of each node
        #unique_list = find_unique(cand_list)
        for i, cand_edge in enumerate(cand_list):
            print(str(i) + ' ' + cand_edge.get_triplet())
            #print(cand_edge)
        idx = int(input('which one? '))
        print('grounding achieved!')
        print(cand_list[idx].get_triplet())
        display_bbox(cand_list[idx].sub)
else:
    # no grounding, ask again
    print('ask again (no grounding)')
            
