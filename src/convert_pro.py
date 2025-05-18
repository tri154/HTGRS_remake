import os
from tqdm import tqdm
import ujson as json
import numpy as np
from adj_utils import sparse_mxs_to_torch_sparse_tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer
import scipy.sparse as sp


cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def read_docred_con(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0
    maxlen = 0
    entity_number = []
    entity_1 =[]
    entity_2 = []
    entity_3 = []
    entity_4 = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        # doc_path = preprocess_data(sample)
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        mention_pos = []
        ent_num = len(entities)    
        men_num = 0
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((pos[0]))
                entity_end.append((pos[1]))
                mention_pos.append((pos[0], pos[1]))
                men_num += 1

        sent_pos = {}
        i_t = 0
        i_s = 0
        sent_map = {}
        for sent in sample['sents']:
            sent_pos[i_s] = len(sents)
            for token in sent:
                tokens_wordpiece = tokenizer.tokenize(token)
                for start, end in mention_pos:
                    if start == i_t:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if end == i_t + 1:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                i_t += 1
            sent_map[i_t] = len(sents)
            i_s += 1
        sent_pos[i_s] = len(sents)


        sentl_node = []
        for l in range(len(sent_pos) - 1):
            sentl_node += [[l, l, l, l, l,l+ent_num+men_num, 2]]
        mention_pos = []
        sentl_pos = []
        for i in range(len(sent_pos) - 1):
            sentl_pos.append((sent_pos[i], sent_pos[i + 1]))

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                r = label['r']
                dist = label['dist']
                if (label['h'], label['t']) not in train_triple:
                    #train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]
                else:
                    #train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]

        entity_pos = []
        men_id = 0
        for e_id, e in enumerate(entities):
            entity_pos.append([])
            for m in e:
                start = sent_map[m["pos"][0]]
                end = sent_map[m["pos"][1]]
                s_id = m["sent_id"]

                entity_pos[-1].append((start, end, e_id, s_id, men_id, men_id+ent_num))
                men_id += 1

        entity_node = []
        mention_node = []
        for idx in range(len(entity_pos)):
            entity_node += [[idx, idx, idx, idx, idx, idx, 0]]
            for item in entity_pos[idx]:
                mention_node += [list(item) + [1]]

        nodes = entity_node + mention_node + sentl_node
        nodes = np.array(nodes)

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
        adjacency[3] = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1,adjacency[3])

        # adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])
        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(4)])


        relations, hts, dists = [], [], []
        for h, t in train_triple.keys():
            #relation = [0] * len(docred_rel2id)
            relation = [0] * len(cdr_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                #evidence = mention["evidence"]
                if mention["dist"] == "CROSS":
                    dist =1
                elif mention["dist"] == "NON-CROSS":
                    dist = 0
            relations.append(relation)
            hts.append([h, t])
            dists.append(dist)
            # pos_samples += 1

        maxlen = max(maxlen, len(sents))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        if len(entity_pos) > max_entity:
            max_entity = len(entity_pos)
            if max_entity > 50:
                print(i_line, max_entity)
        if len(hts) > 0:
            feature = {'input_ids': input_ids,
                       'entity_pos': entity_pos,
                       'labels': relations,
                       'hts': hts,
                       'title': sample['title'],
                       'dists': dists,
                       'adjacency': adjacency,
                       'link_pos': sentl_pos,
                       'nodes_info': nodes,
                       }
            features.append(feature)
        #print("hts:", len(feature['hts']))
        entity_number.append(len(entity_pos))
        print("entity_number:", len(entity_pos))
        if 1<=len(entity_pos)<5:
            entity_1.append(sample)
        elif 5<=len(entity_pos)<10:
            entity_2.append(sample)
        elif 10<= len(entity_pos)<15:
            entity_3.append(sample)
        else:
            entity_4.append(sample)
    print("# of documents {}.".format(i_line))
    # print("# of positive examples {}.".format(pos_samples))
    # print("# of negative examples {}.".format(neg_samples))
    print(f'maxlen:{maxlen}')
    print(f'max_entity_num:{max_entity}')
    print(len(entity_1),len(entity_2),len(entity_3),len(entity_4))
    # with open("./dataset/cdr/divide/first.json", "w") as fh:
    #     json.dump(entity_1, fh)
    # with open("./dataset/cdr/divide/second.json", "w") as fh:
    #     json.dump(entity_2, fh)
    # with open("./dataset/cdr/divide/third.json", "w") as fh:
    #     json.dump(entity_3, fh)
    # with open("./dataset/cdr/divide/fourth.json", "w") as fh:
    #     json.dump(entity_4, fh)
    return features


if __name__ == '__main__':
    data_dir = 'dataset\cdr\convert_CDR'
    tokenizer = AutoTokenizer.from_pretrained('./biobert_base')
    #rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), 'r'))
    rel2id = {'1:NR:2': 0, '1:CID:2': 1}
    id2rel = {v: k for k, v in rel2id.items()}
    #word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), 'r'))
    #ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), 'r'))
    train_in_file = os.path.join(data_dir, 'convert_train.json')
    dev_in_file = os.path.join(data_dir, 'convert_dev.json')
    test_in_file = os.path.join(data_dir, 'convert_test.json')
    train_feature = read_docred_con(dev_in_file, tokenizer)
