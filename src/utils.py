import torch
import random
import dgl
import numpy as np
from torch import nn


def weighted_path_score(G, path):
   edges = zip(path, path[1:])
   #edges:[(0,17),(17,44),(44,23),(23,2)]
   return sum(G.edges[u, v].get('weight', 1) for u, v in edges)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)
def add_logits_to_features(features, indices, logits):
    for i, idx in enumerate(indices):
        feature = features[idx][0]
        if 'teacher_logits' in feature:
            feature.pop('teacher_logits')
        feature['teacher_logits'] = logits[i]
        assert logits[i].shape[0] == len(feature['hts'])

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for (f, _) in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for  (f, _) in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for  (f, _) in batch]
    labels = [f["labels"] for  (f, _) in batch]
    # dists = [f["dists"] for f in batch]
    entity_pos = [f["entity_pos"] for  (f, _) in batch]
    hts = [f["hts"] for  (f, _) in batch]
    link_pos = [f["link_pos"] for  (f, _) in batch]       
    adjacency = [f["adjacency"] for  (f, _) in batch]
    nodes_info = [f["nodes_info"] for  (f, _) in batch]           
    nodes_info = [torch.tensor(item, dtype=torch.long) for item in nodes_info]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    # output = (input_ids, input_mask, labels, entity_pos, hts)
    teacher_logits = []
    for (f, _) in batch:
        if "teacher_logits" in f:
            teacher_logits.append(f["teacher_logits"])
        else:
            teacher_logits.append(None)
    input_indices = [idx for (_, idx) in batch]
    output = (input_ids, input_mask, labels, entity_pos, hts, adjacency, link_pos, nodes_info, teacher_logits, input_indices)
    #output = (input_ids, input_mask, labels, entity_pos, hts, adjacency, link_pos, nodes_info, sub_nodes, sub_adjacency)
    return output


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.ignore = ignore

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)
        self.embedding.weight.requires_grad = not freeze

        if pretrained:
            self.load_pretrained(pretrained, mapping)

        self.drop = nn.Dropout(dropout)

    def load_pretrained(self, pretrained, mapping):
        """
        Args:
            weights: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids
            trainable: (bool)

        Returns: updates the embedding matrix with pre-trained embeddings
        """
        # if self.freeze:
        pret_embeds = torch.zeros((self.num_embeddings, self.embedding_dim))
        # else:
        # pret_embeds = nn.init.normal_(torch.empty((self.num_embeddings, self.embedding_dim)))
        for word in mapping.keys():
            if word in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word])
            elif word.lower() in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])
        self.embedding = self.embedding.from_pretrained(pret_embeds, freeze=self.freeze) # , padding_idx=self.ignore

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds
