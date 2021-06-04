"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.data_utils import  domain2id, OP_SET, make_turn_label, postprocessing # make_slot_meta,
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
# from pytorch_transformers import BertTokenizer, BertConfig
from transformers import BertTokenizer, BertConfig
# from transformers import ElectraModel, ElectraTokenizer, PretrainedConfig

from model import SomDST
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    ontology = json.load(open(args.ontology_data))
    slot_meta = json.load(open(args.slot_meta))
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", additional_special_tokens = ['[SLOT]', '[NULL]','[EOS]', ' ; '])
    # data = prepare_dataset(os.path.join(args.data_root, args.test_data),
    #                        tokenizer,
    #                        slot_meta, args.n_history, args.max_seq_length, args.op_code)

    model_config = PretrainedConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP_SET[args.op_code]
    model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'])
    # ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    # model.load_state_dict(ckpt)


    model.eval()
    model.to(device)

    if args.eval_all:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, True)
    else:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         args.gt_op, args.gt_p_state, args.gt_gen)


def model_evaluation(model, test_data, tokenizer, slot_meta, epoch, op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False):
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}

    results = {}
    last_dialog_state = {}
    wall_times = []
    for di, i in enumerate(tqdm(test_data)):
        if i.turn_id == 0:
            last_dialog_state = {}

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        
        
        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.LongTensor([i.input_mask]).to(device) # data_utils MultiWozDataset에서는 Long이던데 여기선 why FloatTensor? -> changed into LongTensor
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)
        # print(input_ids)
        # print(input_mask)
        # print(segment_ids)
        # print(state_position_ids)

        d_gold_op, _, _ = make_turn_label(slot_meta, last_dialog_state, i.gold_state,
                                          tokenizer, op_code, dynamic=True) # 여기는 왜 dynamic True일까? test_data_raw에서 바로 불러왔기 때문에 형식이 '도메인-슬롯-밸류'이기 때문에 동적 변환
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device)

        start = time.perf_counter()
        MAX_LENGTH = 9
        with torch.no_grad():
            # ground-truth state operation
            gold_op_inputs = gold_op_ids if is_gt_op else None
            d, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            op_ids=gold_op_inputs) # None
            # print(d)
            # print(s)
            # print(g)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
            # print(g)
        else:
            generated = []


        if is_gt_op:
            pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
        else:
            pred_ops = [id2op[a] for a in op_ids.tolist()]
        gold_ops = [id2op[a] for a in d_gold_op]

        if is_gt_gen:
            # ground_truth generation
            gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
        else:
            gold_gen = {}
        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code, gold_gen)
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        # print(f'pred_state: {pred_state}')
        if set(pred_state) == set(i.gold_state):
            joint_acc += 1
        # key = str(i.id) + '_' + str(i.turn_id)
        key = str(i.id) + '-' + str(i.turn_id)  # 제출형식에 맞춤
        results[key] = [sorted(pred_state), sorted(i.gold_state)]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
        op_acc += temp_acc

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
                final_joint_acc += 1
            final_slot_F1_pred += temp_f1
            final_slot_F1_count += count

        # Compute operation F1 score
        for p, g in zip(pred_ops, gold_ops):
            all_op_F1_count[g] += 1
            if p == g:
                tp_dic[g] += 1
                op_F1_count[g] += 1
            else:
                fn_dic[g] += 1
                fp_dic[p] += 1

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    op_acc_score = op_acc / len(test_data)
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000
    op_F1_score = {}
    for k in op2id.keys():
        tp = tp_dic[k]
        fn = fn_dic[k]
        fp = fp_dic[k]
        precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        op_F1_score[k] = F1

    print("------------------------------")
    print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
          (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Epoch %d op accuracy : " % epoch, op_acc_score)
    print("Epoch %d op F1 : " % epoch, op_F1_score)
    print("Epoch %d op hit count : " % epoch, op_F1_count)
    print("Epoch %d op all count : " % epoch, all_op_F1_count)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    json.dump(results, open('preds_%s.json' % str(epoch).zfill(2), 'w'), ensure_ascii=False)
    per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score,
              'op_acc': op_acc_score, 'op_f1': op_F1_score, 'final_slot_f1': final_slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='data/WizardOfSeoul', type=str)
    parser.add_argument("--train_data", default='new_train_dials.json', type=str)
    parser.add_argument("--dev_data", default='new_dev_dials.json', type=str)
    parser.add_argument("--test_data", default='test_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--slot_meta", default='slot_meta.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab2.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/koelectra_base.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='assets/koelectra_base.bin', type=str)
    parser.add_argument("--save_dir", default='outputs', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=True, action='store_true')

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )

    parser.add_argument("--gt_op", default=False, action='store_true')
    parser.add_argument("--gt_p_state", default=False, action='store_true')
    parser.add_argument("--gt_gen", default=False, action='store_true')
    parser.add_argument("--eval_all", default=False, action='store_true')

    args = parser.parse_args()
    main(args)
