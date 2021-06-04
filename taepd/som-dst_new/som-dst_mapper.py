import json
from tqdm import tqdm
import random
from collections import Counter, defaultdict
from typing import List, Dict
import pdb

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit , KFold


def make_WOS_like_WOZ(file_path='./train_dials.json', is_eval=False, dev_split=0.1, k=0):
    dials = json.load(open(file_path))
    new_dial = []
    for dial_dict in tqdm(dials):    
        transformed_dialogue = {}
        transformed_dialogue['dialogue_idx'] = dial_dict['dialogue_idx']
        transformed_dialogue['domains'] = dial_dict['domains']
        dials = []
        last_bs = []
        # 순서를 맞추기 위해 최초 system dummy turn 추가 
        dialogue = dial_dict['dialogue']
        dialogue.insert(0, {'role': "sys", "text": ""})
        # 마지막 sys turn 삭제
        dialogue.pop()
        last_domain = ""
        for idx, i in enumerate(range(0, len(dialogue), 2)):
            dial = {}  # turn_dialog
            sys_dial = dialogue[i]
            user_dial = dialogue[i+1]
            dial['system_transcript'] = sys_dial["text"]
            dial['turn_idx'] = idx
            dial['belief_state'] = []
            if not is_eval:
                for s in user_dial['state']:
                    slots = []
                    states = s.split('-')
                    slot = states[0]+'-'+states[1] # 관광-숙소
                    value = states[2] # 호텔
                    slots.append([slot, value])
                    dial['belief_state'].append({"slots": slots, "act":"inform"})
                dial['turn_label'] = [bs["slots"][0] for bs in dial['belief_state'] if bs not in last_bs] 
                # domain transition을 고려한 turn domain 
                crnt_domain = get_domain(i, dialogue, last_domain)
                dial['domain'] = crnt_domain
                last_domain = crnt_domain
            else:
                dial['domain'] = "관광"  # dummy domain
            dial['transcript'] = user_dial['text']
            dial['system_acts'] = []          
            last_bs = dial['belief_state']
            dials.append(dial)
        transformed_dialogue['dialogue'] = dials
            # print(transformed_dialogue)
            # break
        new_dial.append(transformed_dialogue)
    if not is_eval:
        train, dev = shuffle_and_split(new_dial, dev_split=dev_split, k=k)
        print(f'train length : {len(train)}')
        print(f'dev length : {len(dev)}')

        with open('./data/train_dials.json', 'w') as f:
            json.dump(train, f, ensure_ascii=False)

        with open('./data/dev_dials.json', 'w') as f:
            json.dump(dev, f, ensure_ascii=False)
    else:
        with open('./data/test_dials.json', 'w') as f:
            json.dump(new_dial, f, ensure_ascii=False)


def get_domain(i, dialogue, last_domain):
    """
    현재 turn dialogue와 이전 turn dialogue의 state의 domain을 count하여
    현재 turn에 새롭게 발생한 domain으로 변경하거나 변화가 없으나 이전 domain 유지
    """
    crnt_user_dial = dialogue[i+1] 
    if i == 0: 
        if crnt_user_dial['state']:                            # 시작은 한 도메인으로 이뤄져있다고 간주
            return crnt_user_dial['state'][0].split('-')[0]   # "택시-출발지-그레이 호텔"
        else:  # state가 비어있는 경우 ex) 안녕하세요. state: []
            return ""
    else:
        pred_user_dial = dialogue[i-1]
        pred_d_list = Counter([s.split('-')[0] for s in pred_user_dial['state']])
        crnt_d_list = Counter([s.split('-')[0] for s in crnt_user_dial['state']])
        diff = crnt_d_list - pred_d_list
        if not diff:
            # Sometimes, metadata is an empty dictionary, bug?  
            if not pred_d_list or not crnt_d_list:
                return ""
            return last_domain
        else:
            return diff.most_common()[0][0]
    



# def shuffle_and_split(data):
#     random.seed(a=42)
#     test_size = 0.1
#     test_length = int(len(data) * 0.1)
#     random.shuffle(data)
#     dev = data[:test_length]
#     train = data[test_length:]
#     return train, dev

def shuffle_and_split(data: List[Dict], dev_split=0.1, k=0):
    num_data = len(data)
    num_dev = int(num_data * dev_split)
    if not num_dev:
        return data, []  # no dev dataset
    
#     # dom_mapper: domain transition 횟수를 기준으로 dialogue_idx를 mapping하는 dict
#     dom_mapper = defaultdict(list)
#     for d in data:
#         dom_mapper[len(d["domains"])].append(d["dialogue_idx"])
    
#     num_per_domain_transition = int(num_dev / len(dom_mapper))

#     # domain 전이횟수 종류를 같은 비율이 되도록 dev_idx 샘플링
#     dev_idx = []
#     # for v in dom_mapper.values():
#     #     idx = random.sample(v, num_per_domain_transition)  # random.sample(seq or set, n) s에서 비복원으로 n개 샘플링
#     #     dev_idx.extend(idx)

#     # domain 전이횟수 비율에 따라 분포하게 dev_idx 샘플링
#     for v in dom_mapper.values():
#         idx = random.sample(v, int(len(v)*0.1))  # random.sample(seq or set, n) s에서 비복원으로 n개 샘플링
#         dev_idx.extend(idx)

    train_data, dev_data = [], []
    
    train_idx, dev_idx = create_skfold_dataset_idx(data, k)

    for i, d in enumerate(data):
        if i in dev_idx:
            dev_data.append(d)
        else:
            train_data.append(d)

    # for d in data:
    #     if d["dialogue_idx"] in dev_idx:
    #         dev_data.append(d)
    #     else:
    #         train_data.append(d)

    return train_data, dev_data


def create_skfold_dataset_idx(data, k):
    # labels
    label = [len(d['domains']) for d in data]
    cv = StratifiedShuffleSplit(n_splits=10, 
                                test_size = 0.1, 
                                train_size= 0.9, 
                                random_state=42)
    print('k ', k)
    for idx , (train_idx , dev_idx) in enumerate(cv.split(data, label)):
        if idx == k:
            print(train_idx[:10])
            return train_idx, dev_idx


if __name__=='__main__':
    make_WOS_like_WOZ('../input/data/train_dataset/train_dials.json', is_eval=False, dev_split=0.1, k=7)
    # make_WOS_like_WOZ('../input/data/eval_dataset/eval_dials.json', is_eval=True)
    # make_WOS_like_WOZ('../eval_dataset/eval_dials.json', is_eval=True)