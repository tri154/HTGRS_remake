import torch
import torch.nn as nn
# from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
from torch.nn.utils.rnn import pad_sequence    
from rgcn import RGCN_Layer                     
from utils import EmbedLayer
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda(0).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CC_module(nn.Module):

    def __init__(self, in_dim=256):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,                                                                                                      1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,                                                                                                     1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=512, num_labels=-1, max_entity=22):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = 512
        self.loss_fnt = ATLoss()
        self.cc_module = CC_module()
        self.extractor_trans = nn.Linear(config.hidden_size, emb_size)
        self.ht_extractor = nn.Linear(emb_size*4, emb_size*2)
        #self.MIP_Linear = nn.Linear(emb_size * 7, emb_size * 5)
        self.MIP_Linear = nn.Linear(emb_size * 6, emb_size * 5)
        self.MIP_Linear1 = nn.Linear(emb_size * 5, emb_size * 4)
        self.MIP_Linear2 = nn.Linear(emb_size * 4, emb_size * 2)
        self.linear = nn.Linear(emb_size * 4, 1)
        self.MIP_Linear3 = nn.Linear(emb_size * 2, emb_size)
        self.softmax = nn.Softmax(dim=1)
        self.bilinear = nn.Linear(emb_size * 2, config.num_labels)
        self.emb_size = emb_size
        self.num_labels = num_labels
        self.max_entity = max_entity                                                                   
        self.type_dim = 20           
        self.rgcn = RGCN_Layer(emb_size + self.type_dim, emb_size, 1, 4)
        self.type_embed = EmbedLayer(num_embeddings=3, embedding_dim=self.type_dim, dropout=0.0)       
        inter_channel = int(emb_size // 2)
        self.sigmoid = nn.Sigmoid()
        self.conv_reason_e_l1 = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_reason_e_l2 = nn.Sequential(nn.Conv2d(inter_channel, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),)
        self.conv_reason_e_l3 = nn.Sequential(
            nn.Conv2d(inter_channel, emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(inplace=True), )

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def make_graph(self, sequence_output, attention, entity_pos, link_pos, nodes_info):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()   #[batch_size, num_heads, sequence_length, sequence_length]
        nodes_batch = []
        new_nodes_batch = []
        entity_att_batch = []
        entity_node_batch = []
        mention_pos_batch = []
        mention_att_batch = []
        for i in range(len(entity_pos)):    
            entity_nodes, mention_nodes, link_nodes = [], [], []
            entity_att = []
            mention_att = []
            mention_pos = []
            for start, end in link_pos[i]:
                if end + offset < c:
                    link_rep = sequence_output[i, start + offset: end + offset]
                    link_att = attention[i, :, start + offset: end + offset, start + offset: end + offset]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                elif start + offset < c:
                    link_rep = sequence_output[i, start + offset:]
                    link_att = attention[i, :, start + offset:, start + offset:]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                else:
                    link_rep = torch.zeros(self.hidden_size).to(sequence_output)
                link_nodes.append(link_rep)
            for e in entity_pos[i]:
                mention_pos.append(len(mention_att))
                if len(e) > 1:
                    m_emb, e_att = [], []
                    for start, end, e_id, sid, mid, nid in e:
                    #for start, end, e_id, h_lid, t_lid, sid in e:
                        if start + offset < c:
                            mention_nodes.append(sequence_output[i, start + offset])    
                            m_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            mention_att.append(attention[i, :, start + offset])
                        else:
                            mention_nodes.append(torch.zeros(self.hidden_size).to(sequence_output))
                            m_emb.append(torch.zeros(self.hidden_size).to(sequence_output))
                            e_att.append(torch.zeros(h, c).to(attention))
                            mention_att.append(torch.zeros(h, c).to(attention))
                    if len(m_emb) > 0:
                        m_emb = torch.logsumexp(torch.stack(m_emb, dim=0), dim=0) 
                        e_att = torch.stack(e_att, dim=0).mean(0)   
                    else:
                        m_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    #start, end, e_id, h_lid, t_lid, sid = e[0]
                    start, end, e_id, sid, mid, nid = e[0]
                    if start + offset < c:
                        mention_nodes.append(sequence_output[i, start + offset])
                        m_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                        mention_att.append(attention[i, :, start + offset])
                    else:
                        m_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        mention_att.append(torch.zeros(h, c).to(attention))
                entity_nodes.append(m_emb) 
                entity_att.append(e_att)    
            mention_pos.append(len(mention_att))
            entity_att = torch.stack(entity_att, dim=0)
            entity_att_batch.append(entity_att) ###
            entity_nodes = torch.stack(entity_nodes, dim=0)
            mention_nodes = torch.stack(mention_nodes, dim=0)
            mention_att = torch.stack(mention_att, dim=0)
            link_nodes = torch.stack(link_nodes, dim=0)
            nodes = torch.cat([entity_nodes, mention_nodes, link_nodes], dim=0)
            #print("node:", nodes.shape)
            nodes_type = self.type_embed(nodes_info[i][:, 6].to(sequence_output.device))
            nodes = torch.cat([nodes, nodes_type], dim=1)  
            nodes_batch.append(nodes) 
            #print(nodes.shape)
            entity_node_batch.append(entity_nodes) 
            mention_att_batch.append(mention_att)  
            mention_pos_batch.append(mention_pos)  
        nodes_batch = pad_sequence(nodes_batch, batch_first=True, padding_value=0.0)    
        return nodes_batch, entity_att_batch, entity_node_batch, mention_att_batch, mention_pos_batch

    def relation_map(self, gcn_nodes, entity_att, entity_pos, sequence_output, mention_att):
        entity_s, mention_s = [], []
        entity_c, mention_c = [], []
        ec_rep = [] 
        nodes = gcn_nodes[-1]  
        m_num_max = 0
        e_num_max = 0
        for i in range(len(entity_pos)):
            m_num, _, _ = mention_att[i].size()
            m_num_max = m_num if m_num > m_num_max else m_num_max
            e_num = len(entity_pos[i])
            e_num_max = e_num if e_num > e_num_max else e_num_max
        for i in range(len(entity_pos)):
            e_num = len(entity_pos[i])
            entity_stru = nodes[i][: e_num] 
            m_num, head_num, dim = mention_att[i].size()
            mention_stru = nodes[i][e_num: e_num+m_num] 
            e_att = entity_att[i].mean(1)
            e_att = e_att / (e_att.sum(1, keepdim=True) + 1e-5)
            e_context = torch.einsum('ij, jl->il', e_att, sequence_output[i])  
            m_att = mention_att[i].mean(1)
            m_att = m_att / (m_att.sum(1, keepdim=True) + 1e-5) 
            m_context = torch.einsum('ij,jl->il', m_att, sequence_output[i])  
            # print(entity_stru.size())
            n, h = entity_stru.size()   
            e_s = torch.zeros([e_num_max, h]).to(sequence_output)
            e_s[:n] = entity_stru
            e_s_map = torch.einsum('ij, jk->jik', e_s, e_s.t()) 
            entity_s.append(e_s_map)
            m, h = mention_stru.size()
            m_s = torch.zeros([m_num_max, h]).to(sequence_output)
            m_s[:m] = mention_stru
            m_s_map = torch.einsum('ij,jk->jik', m_s, m_s.t())  
            mention_s.append(m_s_map)
            n, h_2 = e_context.size()
            e_c = torch.zeros([e_num_max, h_2]).to(sequence_output)
            e_c[:n] = e_context
            ec_rep.append(e_c)
            e_c_map = torch.einsum('ij, jk->jik', e_c, e_c.t()) 
            entity_c.append(e_c_map)
            m, h = m_context.size() 
            m_c = torch.zeros([m_num_max, h]).to(sequence_output)   
            m_c[:m] = m_context
            m_c_map = torch.einsum('ij,jk->jik', m_c, m_c.t())  
            mention_c.append(m_c_map)
        ec_rep = torch.stack(ec_rep, dim=0)
        entity_c = torch.stack(entity_c, dim=0)
        entity_s = torch.stack(entity_s, dim=0)
        mention_c = torch.stack(mention_c, dim=0)
        mention_s = torch.stack(mention_s, dim=0)
        return entity_c, entity_s, mention_c, mention_s, ec_rep



    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                adjacency=None,         
                link_pos=None,      
                nodes_info=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        sequence_output = self.extractor_trans(sequence_output)
        # print("a:", sub_nodes[0].shape)
        # print("b:", sub_nodes[1].shape)
        # print("labels:", len(labels[0]), len(labels[1]))
        nodes, entity_att, entity_node_batch, mention_att, mentions_pos = self.make_graph(sequence_output, attention, entity_pos, link_pos, nodes_info)
        gcn_nodes = self.rgcn(nodes, adjacency)
        entity_c, entity_s, mention_c, mention_s, ec_rep = self.relation_map(gcn_nodes, entity_att, entity_pos, sequence_output, mention_att)
        #relation segementation
        r_rep_e = self.conv_reason_e_l1(entity_s) #[batch_size, inter_channel, ent_num, ent_num]
        cc_output = self.cc_module(r_rep_e)
        r_rep_e_2 = self.conv_reason_e_l2(cc_output)
        cc_output_2 = self.cc_module(r_rep_e_2)
        r_rep_e_3 = self.conv_reason_e_l3(cc_output_2)
        # cc_output_3 = self.cc_module(r_rep_e_3)
        relation = []
        entity_h = []
        entity_t = []
        eh = []
        et = []
        e_tou = []
        e_wei = []
        sc_feature_e = []
        nodes_re = torch.cat([gcn_nodes[0], gcn_nodes[-1]], dim=-1)
        #print("nodes_re:", nodes_re.shape)
        for i in range(len(entity_pos)):    
            #print("hts:", hts[i])
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device) 
            # print("ht_i:", ht_i)  
            r_v1 = r_rep_e_3[i, :, ht_i[:, 0],ht_i[:, 1]].transpose(1, 0)
            relation.append(r_v1)   
            e_h = torch.index_select(nodes_re[i], 0, ht_i[:, 0])   
            e_t = torch.index_select(nodes_re[i], 0, ht_i[:, 1])    
            # print("e_h:", e_h.shape)
            # print("e_t:", e_t.shape)
            entity_h.append(e_h)
            entity_t.append(e_t)
            eh = torch.index_select(ec_rep[i], 0, ht_i[:, 0])
            et = torch.index_select(ec_rep[i], 0, ht_i[:, 1])
            e_tou.append(eh)
            e_wei.append(et)
        relation = torch.cat(relation, dim=0)
        entity_h = torch.cat(entity_h, dim=0)
        entity_t = torch.cat(entity_t, dim=0)
        # print("entity_h:", entity_h.shape)
        # print("entity_t:", entity_t.shape)
        entity_ht = self.ht_extractor(torch.cat([entity_h, entity_t], dim=-1)) 
        # print("entity_ht:", entity_ht.shape)
        e_tou = torch.cat(e_tou, dim=0)
        e_wei = torch.cat(e_wei, dim=0)
        e_tw = torch.cat([e_tou, e_wei], dim=-1)
        relation_rep = torch.cat([relation, e_tw, entity_ht], dim=-1)
        relation_rep = self.MIP_Linear1(relation_rep)  
        relation_rep = torch.tanh(self.MIP_Linear2(relation_rep))
        logits = self.bilinear(relation_rep)
        # logits = self.bilinear(torch.tanh(enhanced_features))
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            a = loss.to(sequence_output)
            output = (a,) + output
            # print("output:", output)
        return output
