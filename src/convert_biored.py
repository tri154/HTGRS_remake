import os
from tqdm import tqdm
import ujson as json
import numpy as np
from adj_utils import sparse_mxs_to_torch_sparse_tensor
from transformers import AutoTokenizer
import re
import scipy.sparse as sp
# from subGraph import preprocess_data


# docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}
bio_rel2id = {'Na':0, 'Association': 1, 'Positive_Correlation': 2, 'Negative_Correlation': 3, 'Bind': 4, 'Drug_Interaction': 5, 'Cotreatment': 6, 'Comparison': 7, 'Conversion': 8}


def read_biored(file_in, tokenizer, max_seq_length = 1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0
    maxlen = 0
    entity_num = []
    entity_1 = []
    entity_2 = []
    entity_3 = []
    entity_4 = []
    if file_in == "":
        return None

    with open(file_in, "r") as fh:
        data = json.load(fh)
    """
    'input_ids': input_ids,
    'entity_pos': entity_pos,
    'labels': relations,
    'hts': hts,
    'title': sample['title'],
    'dists': dists,
    """
    #document = data["documents"]
    #for sample in tqdm(document, desc="Example"):
    for sample in tqdm(data, desc="Example"):
        if len(sample["relations"]) == 0:
            continue
        sent = ''
        sentsss = ''
        sent_map = []
        entity_id = []
        mention_pos = []
        men_ent_list = []
        for i in range(50):
            men_ent_list.append([])
            mention_pos.append([])
        relations = []
        train_triples = {}
        hts = []
        pmid = int(sample["id"])
        for id, text in enumerate(sample["passages"]):
            sent += text["text"]
        sents = [t.split(' ') for t in sent.split('|')]
        sents[-1] = [t for t in sents[-1] if t != '']
        sentss = [t for t in sent.split('|')]
        for t in sentss:
            sentsss = ' '.join([sentsss,t])
        sentsss = sentsss.strip()
        sentssss = sentsss.split()
        total_len = 0
        for i in range(len(sents)):
            length = len(sents[i])
            sent_map.append([total_len, total_len+length])
            total_len += length
        entity_number = 0
        men_ent_list1 = []
        for i in range(50):
            men_ent_list1.append([])
        for id, text in enumerate(sample["passages"]):
            for index, men in enumerate(text["annotations"]):
                men_text = men["text"]
                men_words = men_text.split()
                men_id = int(men["id"])
                men_ent_id = men["infons"]["identifier"]
                men_ent_type = men["infons"]["type"]
                if ',' in men_ent_id:
                    men_ent_ids = men_ent_id.split(",")
                    for id in men_ent_ids:
                        if id not in entity_id:
                            entity_id.append(id)
                            ent_id = entity_id.index(id)
                            men_ent_list1[ent_id].append(men_id)
                        else:
                            ent_id = entity_id.index(id)
                            men_ent_list1[ent_id].append(men_id)
                else:
                    if men_ent_id not in entity_id:
                        entity_id.append(men_ent_id)
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list1[ent_id].append(men_id)
                    else:
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list1[ent_id].append(men_id)
        men_ent_list1 = [t for t in men_ent_list1 if t != []]
        entity_number = len(men_ent_list1)
        docu = ''
        men_text_occur = {}
        search_pos = 0
        men_index = 0
        for id, text in enumerate(sample["passages"]):
            for index, men in enumerate(text["annotations"]):
                men_text = men["text"]
                men_words = men_text.split()
                men_id = int(men["id"])

                men_ent_id = men["infons"]["identifier"]
                men_ent_type = men["infons"]["type"]
                if ',' in men_ent_id:
                    men_ent_ids = men_ent_id.split(",")
                    for id in men_ent_ids:
                        if id not in entity_id:
                            entity_id.append(id)
                            ent_id = entity_id.index(id)
                            men_ent_list[ent_id].append(men_id)
                        else:
                            ent_id = entity_id.index(id)
                            men_ent_list[ent_id].append(men_id)
                else:
                    if men_ent_id not in entity_id:
                        entity_id.append(men_ent_id)
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list[ent_id].append(men_id)
                    else:
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list[ent_id].append(men_id)
                a = True
                while a:
                    men_start = sentssss.index(men_words[0], search_pos)
                    men_ending = men_start + len(men_words) - 1
                    if sentssss[men_ending] == men_words[-1]:
                        men_end = men_start + len(men_words)
                        a = False
                    else:
                        search_pos = men_start + 1 
                for pos in sent_map:
                    if men_start >= pos[0] and men_end < pos[1] and sent_map.index(pos) <= len(sent_map) - 1:
                        men_sen_id = sent_map.index(pos)
                        break
                    elif sent_map.index(pos) == len(sent_map) - 1:
                        print("men_words:",men_words)
                        print("sentssss:", sentsss)
                        print("start,pos:", men_start, men_end)
                        print("sent_map:", sent_map)
                        print("pos:", pos)
                        print("sent_map_len, pos:", len(sent_map), sent_map.index(pos))
                        print("can not find sentence id of mention")
                        return 0
                    else:
                        continue
                if ',' in men_ent_id:
                    for id in men_ent_ids:
                        ent_id = entity_id.index(id)
                        mention_pos[ent_id].append((men_start, men_end, ent_id, men_sen_id, men_index, men_index+entity_number))
                else:
                    ent_id = entity_id.index(men_ent_id)
                    mention_pos[ent_id].append((men_start, men_end, ent_id, men_sen_id, men_index, men_index+entity_number))
                search_pos = men_end
                men_index += 1

        men_ent_list = [t for t in men_ent_list if t != []]
        mention_pos = [x for x in mention_pos if x != []]
        entity_pos = []
        entity_node = []
        mention_node = []
        for i in range(len(mention_pos)):
            entity_node +=[[i,i,i,i,i,i,0]]
            for men in mention_pos[i]:
                entity_pos.append(men)
                mention_node += [list(men) + [1]]
        new_sents = []
        sent1_map = {}
        i_t = 0
        i_s = 0
        sent1_pos = {}
        for sent in sents:
            sent1_pos[i_s] = len(new_sents)
            for token in sent:
                tokens_wordpiece = tokenizer.tokenize(token)
                for start, end, ent_id, sen_id, men_id, node_index in entity_pos:
                    if i_t == start:
                        tokens_wordpiece = ["^"] + tokens_wordpiece
                    if i_t + 1 == end:
                        tokens_wordpiece = tokens_wordpiece + ["^"]
                sent1_map[i_t] = len(new_sents)
                new_sents.extend(tokens_wordpiece)
                i_t += 1
            sent1_map[i_t] = len(new_sents) 
            i_s += 1
        sent1_pos[i_s] = len(new_sents) 
        sents = new_sents
        sentl_node = []
        ent_num = len(mention_pos)
        men_num = len(entity_pos)
        for l in range(len(sent1_pos) - 1):
            sentl_node += [[l, l, l, l, l, l + ent_num + men_num, 2]]
        sentl_pos = []
        for i in range(len(sent1_pos) - 1):
            sentl_pos.append((sent1_pos[i], sent1_pos[i + 1]))
        nodes = entity_node + mention_node + sentl_node
        nodes = np.array(nodes)
        # xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
        l_sid, r_sid = nodes[xv, 3], nodes[yv, 3]

        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0,
                           'i')  
              adjacency = np.full((4, l_type.shape[0], r_type.shape[0]), 0.0)

        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])

        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])

        adj_temp = np.where((l_type == 1) & (r_type == 2) & (l_sid == r_sid), 1, adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & (l_sid == r_sid), 1, adjacency[2])
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[2])

        adj_temp = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1, adjacency[3])

        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(4)])

        for id, label in enumerate(sample["relations"]):
            # if len(sample["relations"])==0:
            #     print("no relation instances")
            #     continue
            infons = label["infons"]
            ent_1 = infons["entity1"]
            ent_2 = infons["entity2"]
            relation = [0] * len(bio_rel2id)
            r = bio_rel2id[infons["type"]]
            relation[r] = 1
            ent_1_id = entity_id.index(ent_1)
            ent_2_id = entity_id.index(ent_2)
            if (ent_1_id, ent_2_id) not in train_triples:
                train_triples[(ent_1_id, ent_2_id)] = [{'relation': r}]
            else:
                train_triples[(ent_1_id, ent_2_id)].append({'relation': r})
            hts.append([ent_1_id, ent_2_id])
            relations.append(relation)
            pos_samples += 1


        max_entity = max(max_entity, len(mention_pos))
        # print(max_entity)
        maxlen = max(maxlen, len(sents))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        feature = {
                   'input_ids': input_ids,
                   'entity_pos': mention_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': pmid,
                   'adjacency': adjacency,
                   'link_pos': sentl_pos,
                   'nodes_info': nodes,
                   }
        features.append(feature)
        entity_num.append(len(mention_pos))
        print("entity_number:", len(mention_pos))
        if 1 <= len(mention_pos) < 7:
            entity_1.append(sample)
        elif 7 <= len(mention_pos) < 14:
            entity_2.append(sample)
        elif 14 <= len(mention_pos) < 21:
            entity_3.append(sample)
        else:
            entity_4.append(sample)
    # with open("./dataset/biored/divide/first.json", "w") as fh:
    #     json.dump(entity_1, fh)
    # with open("./dataset/biored/divide/second.json", "w") as fh:
    #     json.dump(entity_2, fh)
    # with open("./dataset/biored/divide/third.json", "w") as fh:
    #     json.dump(entity_3, fh)
    # with open("./dataset/biored/divide/fourth.json", "w") as fh:
    #     json.dump(entity_4, fh)
    print("features length:", len(features))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    print(len(entity_1),len(entity_2),len(entity_3),len(entity_4))
    # print(sum(entity_num)/len(entity_num))
    return features



if __name__ == '__main__':
    data_dir = './dataset/biored/Dev.BioC_modified.JSON'
    # data_dir = './dataset/biored/Test.BioC_modified.JSON'
    tokenizer = AutoTokenizer.from_pretrained('./biobert_base')
    #rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), 'r'))
    rel2id = {'1:NR:2': 0, '1:CID:2': 1}
    id2rel = {v: k for k, v in rel2id.items()}
    #word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), 'r'))
    #ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), 'r'))
    train_feature = read_biored(data_dir, tokenizer)
