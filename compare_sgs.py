import imgSG_json_to_graph as ISG
import langSG_json_to_graph as LSG
from sentence_transformers import SentenceTransformer, util
import numpy as np

# word embedding model
model = SentenceTransformer('stsb-roberta-base')

# make SG outputs into graphs
lang_nodes, lang_edges = LSG.create_graph('lang_sg_result.json')
img_nodes, img_edges = ISG.create_graph('custom_prediction.json', 'custom_data_info.json')
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

def ask_questions(curr_leaf_idx):
    global leaf_edges
    global lang_edges
    global ask_list
    global cand_list
    if len(ask_list) == 1:
        print(ask_list[0].get_triplet())
        feedback = input("is this right?(y/n) ")
        if feedback == 'y':
            cand_list.append(ask_list[0])
    else:
        for idx, ask in enumerate(ask_list):
            # list options to human
            print(idx, ask.get_triplet())
            # accept human input
        feedback_idx = input("which one? ")
        selected_edge = ask_list[int(feedback_idx)]
        # if selected edge is the root edge
        if isempty(leaf_edges[curr_leaf_idx].sub.parent):
            # insert the edge in the candidate list
            cand_list.append(selected_edge)
        # if edge has parent edge
        else:
            parent_edge = find_edge(leaf_edges[curr_leaf_idx].sub, lang_edges)
            # next edge to be looked into should be the parent
            leaf_edges.insert(curr_leaf_idx+1, parent_edge)
    # empty ask_list
    ask_list = []

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
    for img_root_edge in img_root_edges:
        if vref.attr is not None:
            # remove img edges with subject attr not matching
            if compare_attr(vref, img_root_edge.sub) < 0.8:
                img_root_edges.remove(img_root_edge)
    #print(img_root_edges)
    if not isempty(img_root_edges):
        cand_list = img_root_edges
    else:
        print('no ', vref.get_name())
    #print(cand_list)
else:
    # "black cat on wooden table"
    # "white plate on wooden table"
    for i, leaf_edge in enumerate(leaf_edges):
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
                    for sub_pred_edge in attr_match:
                        if leaf_edge.sub.attr is not None:
                            if compare_attr(leaf_edge.sub, sub_pred_edge.sub) < 0.8:
                                attr_match.remove(sub_pred_edge)
                    # One matching attribute
                    if len(attr_match) == 1:
                        ask_list.append(attr_match[0])
                        ask_questions(i)
                    # multiple matching attribute
                    elif len(attr_match) > 1:
                        ask_list.extend(attr_match)
                        ask_questions(i)
                    # no matching attribute
                    else:
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
                else:
                # elif len(pred_obj_match) > 1:
                    print('mult obj, no sub match, mult or no pred match')
                    # if there are multiple objects with similar pred
                    # return one with most similar sub
                    sim_sub = []
                    sub_sim_scores = []
                    for cand_sub in pred_obj_match:
                        sim_score = compare_name(leaf_edge.sub, cand_sub.sub)
                        sub_sim_scores.append(sim_score)
                        sim_sub.append(cand_sub.sub.name)
                    sub_max_score_idx = np.argmax(sub_sim_scores)
                    ask_list.append(pred_obj_match[sub_max_score_idx])
                    # ask_list.extend(pred_obj_match)
                    ask_questions(i)
                # This part not needed
                # else:
                #     # ask triplet with most similar pred
                #     #max_score_idx = np.argmax(sim_scores)
                #     #ask_list.append(obj_matching_edges[max_score_idx])
                #     ask_list.extend(pred_obj_match)
                #     ask_questions(i)
        # no matching edge with object match
        else:
            print('no obj match')
            if i < len(leaf_edges)-1:
                # pass if additional leaf nodes exist
                pass
            else:
                # get root node from LSG
                lang_root_edges = LSG.find_edges_with_root(lang_edges)
                # find all edges containing root node in ISG
                img_root_edges = []
                vref = lang_root_edges[0].sub
                for img_edge in img_edges:
                    if compare_name(vref, img_edge.sub) > 0.8:
                        img_root_edges.append(img_edge)
                # match attributes
                for lang_root_edge in lang_root_edges:
                        for img_root_edge in img_root_edges:
                            if lang_root_edge.sub.attr is not None:
                                if compare_attr(lang_root_edge.sub, img_root_edge.sub) < 0.8:
                                    img_root_edges.remove(img_root_edge)
                            # compare predicates
                            if compare_rels(lang_root_edge, img_root_edge) < 0.8:
                                img_root_edges.remove(img_root_edge)
                # ask filtered edges
                ask_list.extend(img_root_edges)
                ask_questions(i)

if not isempty(cand_list):
    if len(cand_list) == 1:
        # achieved grounding
        print('grounding achieved!')
        print(cand_list[0].get_triplet())
    else:
        # multiple candidates
        # ask unique relation of each node
        for i, cand_edge in enumerate(cand_list):
            print(str(i) + ' ' + cand_edge.get_triplet())
        idx = int(input('which one? '))
        print('grounding achieved!')
        print(cand_list[idx].get_triplet())
else:
    # no grounding, ask again
    print('ask again (no grounding)')
                

