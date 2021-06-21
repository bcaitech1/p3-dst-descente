import json
from tqdm import tqdm
import random

def make_WOS_like_WOZ(file_path='./train_dials.json', is_eval=False):
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
            dial['transcript'] = user_dial['text']
            dial['system_acts'] = []
            dial['domain'] = '관광'
            last_bs = dial['belief_state']
            dials.append(dial)
        transformed_dialogue['dialogue'] = dials
            # print(transformed_dialogue)
            # break
        new_dial.append(transformed_dialogue)
    if not is_eval:
        train, dev = shuffle_and_split(new_dial)
        print(f'train length : {len(train)}')
        print(f'dev length : {len(dev)}')

        with open('./new_train_dials_small.json', 'w') as f:
            json.dump(train, f, ensure_ascii=False)

        with open('./new_dev_dials.json', 'w') as f:
            json.dump(dev, f, ensure_ascii=False)
    else:
        with open('./new_eval_dials.json', 'w') as f:
            json.dump(new_dial, f, ensure_ascii=False)

def shuffle_and_split(data):
    random.seed(a=42)
    test_size = 0.1
    test_length = int(len(data) * 0.1)
    random.shuffle(data)
    dev = data[:test_length]
    train = data[test_length:test_length*2]
    return train, dev
    

if __name__=='__main__':
    # make_WOS_like_WOZ('train_dials.json', is_eval=False)
    make_WOS_like_WOZ('../eval_dataset/eval_dials.json', is_eval=True)