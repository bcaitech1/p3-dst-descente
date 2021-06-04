import json
import re

f1 = open('./dh.csv', 'r')
f2 = open('./mh.csv', 'r')
f3 = open('./yd.csv', 'r')
f4 = open('./dk.csv', 'r')
f5 = open('./test_preds_mapped.csv', 'r')

res1 = json.load(f1)
res2 = json.load(f2)
res3 = json.load(f3)
res4 = json.load(f4)
res5 = json.load(f5)



def voting(*args):
    count={}
    for i in args: # list
        try: count[str(sorted(i))] += 1
        except: count[str(sorted(i))]=1
    
    new = sorted(count.items(), key=lambda x: -x[1])
    if new[0][1] == len(args):
        return eval(new[0][0])
    elif new[0][1] == len(args)-1:
        return eval(new[0][0])
    else:
        return args[-1]

def postprocessing(data):
    new_dict = dict()
    for slot, values in data.items():
        lst = []
        for value in values:
            gen = re.sub('\s(?=[\=\(\)\&])|(?<=[\=\(\)\&])\s', "", value)
            lst.append(gen)    
        new_dict[slot] = lst

    return new_dict

res4 = postprocessing(res4) # 도균님 - 꿔바로우 처리 안되어있음.
res_col = [res1, res2, res3, res4, res5]
keys = res1.keys()
poll = {}
for key in keys:
    voted = voting(res1[key], res2[key], res3[key], res4[key], res5[key])
    poll[key] = voted
poll = postprocessing(poll)

with open('./prediction_ensemble_sorted_and_postprocessed_2.csv', 'w') as f:
    json.dump(poll, f, ensure_ascii=False)
