import json
import re

f1 = open('./TRADE_HARDVOTING/dk_trade_7502_no_postprocessing.csv', 'r')
f2 = open('./TRADE_HARDVOTING/trade_predictions_fold_2_7325.csv', 'r')
f3 = open('./TRADE_HARDVOTING/trade_predictions_fold_8_7367.csv', 'r')
f4 = open('./TRADE_HARDVOTING/trade_predictions_fold_5_7433.csv', 'r')
f5 = open('./TRADE_HARDVOTING/trade_predictions_fold_4_7535.csv', 'r')
f6 = open('./SOM-DST_HARDVOTING/som_dst_k2_7637.csv', 'r')
f7 = open('./SOM-DST_HARDVOTING/som_dst_notk_7594.csv', 'r')
f8 = open('./SOM-DST_HARDVOTING/som_dst_k8_8240_7448.csv', 'r')
f9 = open('./SOM-DST_HARDVOTING/som_dst_k3_7665.csv', 'r')
# f10 = open('./SOM-DST_HARDVOTING/som_dst_k4.csv', 'r')

res1 = json.load(f1)
res2 = json.load(f2)
res3 = json.load(f3)
res4 = json.load(f4)
res5 = json.load(f5)
res6 = json.load(f6)
res7 = json.load(f7)
res8 = json.load(f8)
res9 = json.load(f9)
# res10 = json.load(f10)


def voting(*args):
    count={}
    for i in args: # list
        try: count[str(sorted(i))] += 1
        except: count[str(sorted(i))]=1
    
    new = sorted(count.items(), key=lambda x: -x[1])
    if new[0][1] >= len(args)/2 - 2: # 4개 이상이면 그걸로 보팅.
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

res1 = postprocessing(res1) # 도균님 - 꿔바로우 처리 안되어있음.
res_col = [res1, res2, res3, res4, res5, res6, res7, res8, res9]#, res10]
keys = res1.keys()
poll = {}
for key in keys:
    voted = voting(res1[key], res2[key], res3[key], res4[key], res5[key], res6[key], res7[key], res8[key], res9[key])#, res10[key])
    poll[key] = voted
poll = postprocessing(poll)

with open('./ALL_hardvoting_result_2.csv', 'w') as f:
    json.dump(poll, f, ensure_ascii=False)
