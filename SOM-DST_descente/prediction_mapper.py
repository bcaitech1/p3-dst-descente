import json
import re
import argparse
def pred_into_csv(file_path, save_path):
    new_dict = dict()
    with open(file_path, 'r') as f:
        file = json.load(f)
    for k, v in file.items():
        origin, change = k.split('_')[:-1], k.split('_')[-1]
        new_key = '_'.join(origin) + '-' + change
        new_value = []
        for vv in v[0]:
            gen = vv.split('-')[-1]
            gen = re.sub('\s(?=[\=\(\)\&])|(?<=[\=\(\)\&])\s', "", gen)  # =&() 의 문자간 공백 제거
            gen = gen.replace(" : ", ":")
            gen = gen.replace(" , ", ", ")
            new_val = '-'.join(vv.split('-')[:-1])+'-'+gen
            new_value.append(new_val)
        new_dict[new_key] = new_value

    with open(save_path, 'w') as f:
        json.dump(new_dict, fp=f, ensure_ascii=False)

def main(args):
    pred_into_csv(args.prediction_data_dir, args.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_data_dir', default="", type=str)
    parser.add_argument('--save_dir', default="", type=str)
    args = parser.parse_args()
    main(args)