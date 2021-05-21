import argparse
import json
import os
import pickle
import random

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig

from data_utils import (YamlConfigManager, WOSDataset, get_examples_from_dialogues, load_dataset,
                        set_seed, custom_to_mask, custom_get_examples_from_dialogues, )

from evaluation import _evaluation
from inference import inference
from model import TRADE, masked_cross_entropy_for_value
from preprocessor import TRADEPreprocessor
from prettyprinter import cpprint

from pathlib import Path
import glob
import re

import wandb
import time

from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 저장 경로 이름 자동 변경
def increment_output_dir(output_path, exist_ok=False):
    path = Path(output_path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        # 디렉터리에 있는 파일들을 리스트로 만들기 - glob(pathname)
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
    return f"{path}_{n}"


# Get current learning rate
def get_lr(scheduler):
    return scheduler.get_last_lr()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config).values
    cpprint(cfg)


    # random seed 고정
    set_seed(cfg.random_seed)

    # Data Loading
    train_data_file = f"{cfg.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{cfg.data_dir}/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file, cfg.val_ratio)

    train_examples = custom_get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = custom_get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    # Define Preprocessor
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    # Dealing with long texts The maximum sequence length of BERT is 512.
    processor = TRADEPreprocessor(slot_meta, tokenizer, max_seq_length=512, n_gate=cfg.n_gate)

    # Extracting Featrues
    cpprint('Extracting Features...')
    train_features = processor.sep_custom_convert_examples_to_features(train_examples)
    dev_features = processor.sep_custom_convert_examples_to_features(dev_examples)

    # 전체 train data InputFeatur 저장
    with open('custom_train_features.txt', 'wb') as f:
        pickle.dump(train_features, f)
    with open('custom_dev_features.txt', 'wb') as f:
        pickle.dump(dev_features, f)

    # with open('custom_train_features.txt', 'rb') as f:
    #     train_features = pickle.load(f)
    # with open('custom_dev_features.txt', 'rb') as f:
    #     dev_features = pickle.load(f)

    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )
    
    # Model 선언
    config = AutoConfig.from_pretrained('dsksd/bert-ko-small-minimal')
    config.model_name_or_path = 'dsksd/bert-ko-small-minimal'
    config.n_gate = cfg.n_gate
    config.proj_dim = None

    model = TRADE(config, tokenized_slot_meta)

    model.to(device)
    print("Model is initialized")

    # --wandb initialize with configuration
    wandb.init(project='DST', tags=cfg.tag, config=cfg)

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
        num_workers=4,  # num_worker = 4 * num_GPU
        pin_memory=True,
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=cfg.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    print("# dev:", len(dev_data))
    
    # Optimizer 및 Scheduler 선언
    n_epochs = cfg.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    t_total = len(train_loader) * n_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)

    warmup_steps = int(t_total * cfg.warmup_ratio)
    # learning rate decreases linearly from the initial lr set in the optimizer to 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    teacher_forcing = cfg.teacher_forcing_ratio

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    # 모델 저장될 파일 위치 생성
    if not os.path.exists(f"{cfg.model_dir}/{wandb.run.name}"):
        os.mkdir(f"{cfg.model_dir}")
        os.mkdir(f"{cfg.model_dir}/{wandb.run.name}")

    # json data 직렬화
    json.dump(
        vars(cfg),
        open(f"{cfg.model_dir}/{wandb.run.name}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{cfg.model_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    # backward pass시 gradient 정보가 손실되지 않게 하려고 사용(loss에 scale factor를 곱해서 gradient 값이 너무 작아지는 것을 방지)
    scaler = GradScaler()
    best_score, best_checkpoint = 0, 0

    for epoch in range(n_epochs):
        start_time = time.time()
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(device) if not isinstance(b, list) else b for b in batch
            ]
            # mask
            if cfg.mask:
                change_mask_prop = 0.8
                mask_p = random.random()
                if cfg.mask and mask_p < change_mask_prop:
                    input_ids = custom_to_mask(input_ids)

            
            # teacher forcing
            if (
                teacher_forcing > 0.0
                and random.random() < teacher_forcing
            ):
                tf = target_ids
            else:
                tf = None

            optimizer.zero_grad()  # optimizer는 input으로 model parameter를 가진다 -> zero_grad()로 파라미터 컨드롤 가능

            with autocast():  # 밑에 해당하는 코드를 자동으로 mixed precision으로 변환시켜서 실행
                all_point_outputs, all_gate_outputs = model(
                    input_ids, segment_ids, input_masks, target_ids.size(-1), tf
                )


                # generation loss
                loss_1 = loss_fnc_1(
                    all_point_outputs.contiguous(),
                    target_ids.contiguous().view(-1),
                    tokenizer.pad_token_id,
                )

                # gating loss
                loss_2 = loss_fnc_2(
                    all_gate_outputs.contiguous().view(-1, cfg.n_gate),
                    gating_ids.contiguous().view(-1),
                )

                loss = loss_1 + loss_2
            batch_loss.append(loss.item())

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()




            # global_step 추가 부분
            wandb.log({"train/learning_rate": get_lr(scheduler),
                       "train/epoch": epoch
                       })

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}"
                )

                # -- train 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "train/loss": loss.item(),
                    "train/gen_loss": loss_1.item(),
                    "train/gate_loss": loss_2.item(),
                })

        predictions = inference(model, dev_loader, processor, device, cfg.n_gate)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)


        # -- eval 단계에서 Loss, Accuracy 로그 저장
        wandb.log({
            "eval/join_goal_acc": eval_result["joint_goal_accuracy"],
            "eval/turn_slot_f1": eval_result["turn_slot_f1"],
            "eval/turn_slot_acc": eval_result["turn_slot_accuracy"],
        })

        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result['joint_goal_accuracy']:
            cpprint(f"--Update Best checkpoint!, epoch: {epoch+1}")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch
            if not os.path.isdir(cfg.model_dir):
                os.makedirs(cfg.model_dir)
            print("--Saving best model checkpoint")
            torch.save(model.state_dict(), f"{cfg.model_dir}/{wandb.run.name}/best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler.state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'gen_loss': loss_1.item(),
                'gate_loss': loss_2.item(),
            }, os.path.join(f"{cfg.model_dir}/{wandb.run.name}", "training_best_checkpoint.bin"))



        torch.save(model.state_dict(), f"{cfg.model_dir}/{wandb.run.name}/last.pth")
        print(f"time for 1 epoch: {time.time() - start_time}")