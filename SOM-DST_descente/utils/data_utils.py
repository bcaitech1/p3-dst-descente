"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy
from .fix_label import fix_general_label_error

import pdb

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ['관광', '숙소', '식당', '지하철', '택시'] # ["hotel", "train", "restaurant", "attraction", "taxi"]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}


def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state  # {'관광-종류': '공원', '관광-지역': 'dontcare'}
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    keys = list(turn_dialog_state.keys())  # ('관광-종류', '관광-지역')
    for k in keys:
        v = turn_dialog_state[k] # 이번 대화의 slot value  # '공원'
        # print('############')
        # print('v: ', v)
        # print('############')

        if v == 'none':
            turn_dialog_state.pop(k) 
            continue
        vv = last_dialog_state.get(k) # 전 대화에서 가져온 slot value
        # print('vv', vv)
        # pdb.set_trace()  
        try:
            idx = slot_meta.index(k) # slot meta에서 인덱스를 찾는다.
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
            elif vv == v: # 이전 slot value와 같으면 carryover
                op_labels[idx] = 'carryover'
        except ValueError:
            continue
    
    for k, v in last_dialog_state.items():  # {'관광-종류': '공원'}
        vv = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([['[NULL]', '[EOS]'], idx])
        except ValueError:
            continue
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1]) # idx 순서, 즉 slot_meta의 순서를 따라 정렬.
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]
    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                # 서울 중앙 [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] 문제를 해결하기 위해 임시적으로 정규식 적용
                # gen = re.sub('\[UNK\]', "", gen)
                # gen = gen.strip()
                gen = re.sub('\s(?=[\=\(\)\&])|(?<=[\=\(\)\&])\s', "", gen)  # =&() 의 문자간 공백 제거
                gen = gen.replace(" : ", ":")
                gen = gen.replace(" , ", ", ")
                
                last_dialog_state[st] = gen
    return generated, last_dialog_state


# def make_slot_meta(ontology):
#     meta = []
#     change = {}
#     idx = 0
#     max_len = 0
#     for i, k in enumerate(ontology.keys()):
#         d, s = k.split('-')
#         if d not in EXPERIMENT_DOMAINS:
#             continue
#         if 'price' in s or 'leave' in s or 'arrive' in s:
#             s = s.replace(' ', '')
#         ss = s.split()
#         if len(ss) + 1 > max_len:
#             max_len = len(ss) + 1
#         meta.append('-'.join([d, s]))
#         change[meta[-1]] = ontology[k]
#     return sorted(meta), change


def prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, diag_level=False, op_code='4'):
    dials = json.load(open(data_path))
    data = []
    domain_counter = {}
    max_resp_len, max_value_len = 0, 0
    max_line = None

    for dial_dict in dials:
        for domain in dial_dict["domains"]: # domain names
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter.keys():
                domain_counter[domain] = 0
            domain_counter[domain] += 1 # count

        dialog_history = []
        last_dialog_state = {}
        last_uttr = ""
        tmp = 0  # WOS 변환 때문에 임시적으로 추가한 변수
        for ti, turn in enumerate(dial_dict["dialogue"]): # 
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:  # turn domain이 없는 경우 (ex)그냥 인사만 한 상태) "" 빈 문자열 처리할 경우 여기서 누락되는 문제 생김 
                if turn["turn_idx"] == 0:  # 임시 추가 코드   index가 1부터 시작하게 되면서, 새 dialogue처리가 제대로 안됨. last_dialogue_state 를 evaluation에서 초기화 못함                                                     
                    tmp -= 1                # 임시 추가 코드   그렇다고 임의의 도메인을 넣기도 애매하니, 전체적으로 1씩 낮춰줌
                continue                               
            
            turn_id = turn["turn_idx"] + tmp
            turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
            # turn_uttr = (turn["transcript"] + ' ; ' + turn["system_transcript"]).strip()
            dialog_history.append(last_uttr)
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)
            # print(turn_dialog_state)
            last_uttr = turn_uttr
            # print(turn_dialog_state)

            op_labels, generate_y, gold_state = make_turn_label(slot_meta, last_dialog_state,
                                                                turn_dialog_state,
                                                                tokenizer, op_code)
            # print(f'OP_LABELS: {op_labels}')
            # print(f'generate_y : {generate_y}')
            # print(f'gold_state : {gold_state}')
            # print()
            if (ti + 1) == len(dial_dict["dialogue"]):
                is_last_turn = True
            else:
                is_last_turn = False
            
            instance = TrainingInstance(dial_dict["dialogue_idx"], turn_domain,
                                        turn_id, turn_uttr, ' '.join(dialog_history[-n_history:]),
                                        last_dialog_state, op_labels,
                                        generate_y, gold_state, max_seq_length, slot_meta,
                                        is_last_turn, op_code=op_code)

            instance.make_instance(tokenizer)
            data.append(instance)
            last_dialog_state = turn_dialog_state
    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 op_labels,
                 generate_y,
                 gold_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 op_code='4'):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.generate_y = generate_y
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]

    def shuffle_state(self, rng, slot_meta=None):
        new_y = []
        gid = 0
        for idx, aa in enumerate(self.op_labels):
            if aa == 'update':
                new_y.append(self.generate_y[gid])
                gid += 1
            else:
                new_y.append(["dummy"])
        if slot_meta is None:
            temp = list(zip(self.op_labels, self.slot_meta, new_y))
            rng.shuffle(temp)
        else:
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.op_labels, self.slot_meta, new_y, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.op_labels = list(temp[0])
        self.slot_meta = list(temp[1])
        self.generate_y = [yy for yy in temp[2] if yy != ["dummy"]]

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]'):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        # print('max_seq_length: ', max_seq_length)
        # pdb.set_trace()

        state = []  # state 길이만 256을 넘음
        for s in self.slot_meta:
            state.append(slot_token)
            k = s.split('-')
            v = self.last_dialog_state.get(s)
            # print(v)
            if v is not None:
                k.extend(['-', v])
                t = tokenizer.tokenize(' '.join(k))
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)
        # print(f'state:{state}')
        # print(f'max_seq_length : {max_seq_length}')
        avail_length_1 = max_seq_length - len(state) - 3
        # print(f'avail_length_1:{avail_length_1}')

        diag_1 = tokenizer.tokenize(self.dialog_history)
        diag_2 = tokenizer.tokenize(self.turn_utter)
        # print('*****')
        # print(diag_1)
        # print(diag_2)
        # print(max_seq_length)
        # print(len(state))
        # pdb.set_trace()
        # print('*****')

        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]
            # print(diag_1)

        if len(diag_1) == 0 and len(diag_2) > avail_length_1: # 첫 대사인 경우
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]
            # print(diag_2)

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)
        # print('*****')
        # print(diag_1)
        # print(diag_2)
        # print('*****')
        # pdb.set_trace()
        diag = diag_1 + diag_2

        # print(f'diag:{diag}')
        # print(f'len_diag : {len(diag)}')
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag + state
        segment = segment + [1]*len(state)
        self.input_ = input_
        # print(f'X_input:{self.input_}')
        # print(f'X_input_length:{len(self.input_)}')

        self.segment_id = segment
        slot_position = []
        for i, t in enumerate(self.input_):
            if t == slot_token:
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))
        self.input_mask = input_mask
        self.domain_id = domain2id[self.turn_domain]
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = [tokenizer.convert_tokens_to_ids(y) for y in self.generate_y]
    # def __str__(self):
    #     return f'[{self.input_id}], {self.input_mask}, {self.segment_id}, {self.slot_position}, {self.op_ids}, {self.generate_ids}'
        

class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng,
                 ontology, word_dropout=0.1, shuffle_state=False, shuffle_p=0.5):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.shuffle_state = shuffle_state
        self.shuffle_p = shuffle_p
        self.rng = rng

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.shuffle_state and self.shuffle_p > 0.:
            if self.rng.random() < self.shuffle_p:
                self.data[idx].shuffle_state(self.rng, None)
            else:
                self.data[idx].shuffle_state(self.rng, self.slot_meta)
        if self.word_dropout > 0 or self.shuffle_state:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        # for e in batch:
        #     print(len(e.input_id))
        input_ids = torch.LongTensor([f.input_id for f in batch])#, dtype=torch.long)
        input_mask = torch.LongTensor([f.input_mask for f in batch])#, dtype=torch.long)
        segment_ids = torch.LongTensor([f.segment_id for f in batch])#, dtype=torch.long)
        state_position_ids = torch.LongTensor([f.slot_position for f in batch])#, dtype=torch.long)
        op_ids = torch.LongTensor([f.op_ids for f in batch])#, dtype=torch.long)
        domain_ids = torch.LongTensor([f.domain_id for f in batch])#, dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        # max_value error 방지 위해 default=0 추가
        max_update = max([len(b) for b in gen_ids], defalut=0)
        max_value = max([len(b) for b in flatten(gen_ids)], default=0)
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)
        # cnt = 0
        # for b in batch:
        #     if cnt == 1:
        #         continue
        #     with open('./batch_example.txt', 'w') as f:
        #         f.write(f'input_ids : {input_ids}')
        #         f.write('\n')
        #         f.write(f'input_ids shape : {input_ids.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         f.write(f'input_mask  : {input_mask}')
        #         f.write('\n')
        #         f.write(f'input_mask shape : {input_mask.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         f.write(f'segment_ids : {segment_ids}')
        #         f.write('\n')
        #         f.write(f'segment_ids shape : {segment_ids.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         f.write(f'state_position_ids : {state_position_ids}')
        #         f.write('\n')
        #         f.write(f'state_position_ids shape : {state_position_ids.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         f.write(f'op_ids : {op_ids}')
        #         f.write('\n')
        #         f.write(f'op_ids shape : {op_ids.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         f.write(f'domain_ids : {domain_ids}')
        #         f.write('\n')
        #         f.write(f'domain_ids shape : {domain_ids.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         f.write(f'gen_ids : {gen_ids}')
        #         f.write('\n')
        #         f.write(f'gen_ids shape : {gen_ids.shape}')
        #         f.write('\n')
        #         f.write('\n')
        #         cnt = 1

        return input_ids, input_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, max_value, max_update
