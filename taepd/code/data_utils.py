import dataclasses
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import ruamel.yaml
from easydict import EasyDict
import re


def remove_space(text):
    return re.sub('\s(?=[\=\(\)\&])|(?<=[\=\(\)\&])\s', "", text)  # =&() 의 문자간 공백 제거



@dataclass
class OntologyDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    num_turn: int
    target_ids: Optional[List[int]]

@dataclass
class SUMBTInputFeature:
    guid: str                       # dialogue_idx
    input_ids: List[List[int]]      # Tokenized Dialogue context [[r_1, u_1], ...[r_T, u_T]]
    segment_ids: List[List[int]]    # token_type_ids  user_token, padding = 0, system_token = 1
    num_turn: int                   # T (해당 대화의 총 턴 길이)
    target_ids: Optional[List[List[int]]]  # T * J (# of Slot Meta) (각 Slot 별로 Candidates의 ground-truth index)

@dataclass
class OpenVocabDSTFeature:
    guid: str                       # dialogue_idx + turn_idx  (dialogue_level=False 이므로 한 turn씩 append)
    input_id: List[int]             # tokenized dialogue context  [r_1, u_1, ...r_T, u_T]
    segment_id: List[int]           # token_type_ids(Optional) (Bert모델에서만 사용)
    gating_id: List[int]            # J(# of Slot Meta)
    target_ids: Optional[Union[List[int], List[List[int]]]]  # J * N (N: 각 Slot별로 tokenized target value)


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


def load_dataset(dataset_path, dev_split=0.1):
    data = json.load(open(dataset_path))
    num_data = len(data)
    num_dev = int(num_data * dev_split)
    if not num_dev:
        return data, []  # no dev dataset

    # dom_mapper: domain transition 횟수를 기준으로 dialogue_idx를 mapping하는 dict
    dom_mapper = defaultdict(list)
    for d in data:
        dom_mapper[len(d["domains"])].append(d["dialogue_idx"])

    num_per_domain_transition = int(num_dev / len(dom_mapper))  # 3은 domain 전이횟수 종류에 대한 hard coding?
    # domain 전이횟수 종류를 같은 비율이 되도록 dev_idx 샘플링
    dev_idx = []
    for v in dom_mapper.values():
        idx = random.sample(v, num_per_domain_transition)  # random.sample(seq or set, n) s에서 비복원으로 n개 샘플링
        dev_idx.extend(idx)

    train_data, dev_data = [], []
    for d in data:
        if d["dialogue_idx"] in dev_idx:
            dev_data.append(d)
        else:
            train_data.append(d)

    # turn 마다 state를 label화 하기 위한 작업
    dev_labels = {}
    for dialogue in dev_data:
        d_idx = 0
        guid = dialogue["dialogue_idx"]
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            state = turn.pop("state")

            guid_t = f"{guid}-{d_idx}"
            d_idx += 1

            dev_labels[guid_t] = state

    return train_data, dev_data, dev_labels

 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU


def split_slot(dom_slot_value, get_domain_slot=False):
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def build_slot_meta(data):
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue

            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def convert_state_dict(state):
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic


@dataclass
class DSTInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_examples_from_dialogue(dialogue, user_first=False):
    guid = dialogue["dialogue_idx"]
    examples = []
    history = []
    d_idx = 0
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")
        context = deepcopy(history)
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]
        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
            )
        )
        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1
    return examples


def get_examples_from_dialogues(data, user_first=False, dialogue_level=False):
    examples = []
    for d in tqdm(data):
        example = get_examples_from_dialogue(d, user_first=user_first)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples


class DSTPreprocessor:
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()  # 정수형으로 바꿈 (0. -> 0)
            m = torch.cat([array, pad], -1)  # [45, max_length], 마지막 차원을 늘리면서(max_length) 텐서 연결
            new_arrays.append(m.unsqueeze(0))  # [1, 45, max_length], 특정 위치에 1인 차원을 추가

        return torch.cat(new_arrays, 0)  # [batch_size, 45, max_length], list원소들을 합쳐서  tensor로
    # abstract method
    def _convert_example_to_feature(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def recover_state(self):
        raise NotImplementedError

        
# Set Config
class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()

    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(ruamel.yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()

    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                ruamel.yaml.dump(dict(self.values), f)


def custom_to_mask(input_ids):
    SEP_ID = 3
    MASK_ID = 4
    for input_id in input_ids:
        sep_idx = (input_id == SEP_ID).nonzero(as_tuple=False)[1]
        if sep_idx - 3 > 4:
            # for i in range(len(current_id)):
            mask_idxs = set()
            while len(mask_idxs) <= 3:
                rand_idx = random.randrange(2, sep_idx)
                if rand_idx == sep_idx:
                    continue
                mask_idxs.add(rand_idx)

            for mask_idx in list(mask_idxs):
                input_id[mask_idx] = MASK_ID

    return input_ids




def custom_get_example_from_dialogue(dialogue, user_first=False):
    guid = dialogue["dialogue_idx"]
    examples = []
    history = []
    d_idx = 0
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")
        context = deepcopy(history)
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, " # ", user_utter, " * "]
        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
            )
        )
        history.append(sys_utter)
        history.append(" # ")
        history.append(user_utter)
        history.append(" * ")
        d_idx += 1
    return examples


def custom_get_examples_from_dialogues(data, user_first=False, dialogue_level=False):
    examples = []
    for d in tqdm(data):
        example = custom_get_example_from_dialogue(d, user_first=user_first)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples


def get_value(slot, value):
    cadidate = name_ontology[slot]
    for v in cadidate:
        if set(''.join(value.split())) == set(''.join(v.split())):
            return v
    return value