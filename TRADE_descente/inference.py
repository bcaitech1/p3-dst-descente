import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer, AutoConfig, BertTokenizerFast

from data_utils import (WOSDataset, get_examples_from_dialogues, YamlConfigManager)
from model import TRADE
from preprocessor import TRADEPreprocessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference(model, eval_loader, processor, device, n_gate):
    model.eval()
    predictions = {}

    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            # all_point_outputs, all_gate_outputs
            o, g = model(input_ids, segment_ids, input_masks, 9)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config).values

    DATA_DIR = '../input/data/eval_dataset'
    OUTPUT_DIR = f'./results/{args.save_dir}'
    MODEL_DIR = OUTPUT_DIR + "/best.pth"

    # Data Loading
    eval_data = json.load(open(f'{DATA_DIR}/eval_dials.json', 'r'))
    slot_meta = json.load(open(f'{cfg.data_dir}/slot_meta.json'))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    # Dealing with long texts The maximum sequence length of BERT is 512.
    processor = TRADEPreprocessor(slot_meta, tokenizer, max_seq_length=512, n_gate=cfg.n_gate)

    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )

    # Extracting Featrues
    eval_features = processor.convert_examples_to_features(eval_examples)
    eval_features = processor.custom_convert_examples_to_features(eval_examples)

    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=cfg.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# eval:", len(eval_data))

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
    ckpt = torch.load(MODEL_DIR, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    predictions = inference(model, eval_loader, processor, device, cfg.n_gate)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    json.dump(
        predictions,
        open(f"{OUTPUT_DIR}/predictions.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
