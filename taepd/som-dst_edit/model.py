"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers import BertModel, BertTokenizer, ElectraModel, ElectraTokenizer, PreTrainedModel


class SomDST(PreTrainedModel):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(SomDST, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(config, n_op, n_domain, update_id, exclude_domain)
        self.decoder = Decoder(config, self.encoder.bert.embeddings.word_embeddings.weight)
        self.init_weights

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                max_value, op_ids=None, max_update=None, teacher=None):

        ######## ENCODER OUTPUT ########
        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask,
                                   op_ids=op_ids,
                                   max_update=max_update)

        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs
        
        # print('domain_scores:' ,domain_scores.isnan().any())
        # print('state_scores:' ,state_scores.isnan().any())
        # print('decoder_input:' , decoder_inputs.isnan().any())
        # print('seqeunce_output:' , sequence_output.isnan().any())
        # print('pooled_output:' , pooled_output.isnan().any())


        ######## DECODER OUTPUT ########
        gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
                                  pooled_output, max_value, teacher)

        # print('gen_scores:' , gen_scores.isnan().any())
        return domain_scores, state_scores, gen_scores


class Encoder(nn.Module):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel.from_pretrained('dsksd/bert-ko-small-minimal')
        self.tokenizer = BertTokenizer.from_pretrained('dsksd/bert-ko-small-minimal', additional_special_tokens = ['[SLOT]', '[NULL]','[EOS]'])
        self.bert.resize_token_embeddings(len(self.tokenizer)) 
        # self.bert = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        # self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", additional_special_tokens = ['[SLOT]', '[NULL]','[EOS]', ' ; '])
        # self.bert.resize_token_embeddings(len(self.tokenizer)) # 4?
        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, n_op)
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                op_ids=None, max_update=None):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # print('input_ids:' , input_ids.isnan().any())
        # print('token_type_ids:' , token_type_ids.isnan().any())
        # print('attention_mask:' , attention_mask.isnan().any())

        # bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)[0] # Batch x Max_Seq_length x hidden size
        # print(bert_outputs.dtype)
        # print('::::OUTPUT::::')
        # print(bert_outputs)
        # print('::::OUTPUT SHAPE:::')
        # print(bert_outputs.shape)
        # print()
        sequence_output, pooled_output = bert_outputs[:2]
        # print('pooled_output:' , pooled_output.isnan().any())
        # sequence_output = bert_outputs
        # pooled_output = bert_outputs[:, 0, :]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)

        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4
        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        batch_size = state_scores.size(0)

        #### op_id가 주어져 있지 않은 경우 state_scores를 통해 예측 ####
        if op_ids is None:
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1) # state_score 중 softmax 값 가장 높은 op_id
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()
        
        # print(op_ids)
        # print(max_update)

        gathered = []
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            if a.sum().item() != 0: # update할 슬롯이 있다면
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
            else: # update할 슬롯이 없다면
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)
        decoder_inputs = torch.cat(gathered) # update되는 슬롯만 디코더 인풋으로 들어감.
        # print(sequence_output)
        return domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update): # 업데이트 시켜야 하는 슬롯 개수
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                # print(f'w : {w.shape}')
                # print(f'hidden: {hidden.shape}')
                w = self.dropout(w).contiguous()
                # hidden = hidden.view(8,1,768)
                _, hidden = self.gru(w, hidden.contiguous())  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                # attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e4)   # mixed precision 적용 위해
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)