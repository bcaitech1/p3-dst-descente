class DSTEvaluator:
    def __init__(self, slot_meta):
        self.slot_meta = slot_meta
        self.init()

    def init(self):
        self.joint_goal_hit = 0
        self.all_hit = 0
        self.slot_turn_acc = 0
        self.slot_F1_pred = 0
        self.slot_F1_count = 0

    def update(self, gold, pred):
        self.all_hit += 1
        if set(pred) == set(gold):
            self.joint_goal_hit += 1

        # temp_acc = compute_acc(gold, pred, self.slot_meta)
        temp_acc, miss_labels = compute_acc(gold, pred, self.slot_meta)
        self.slot_turn_acc += temp_acc

        temp_f1, _, _, count = compute_prf(gold, pred)
        self.slot_F1_pred += temp_f1
        self.slot_F1_count += count

        return miss_labels

    def compute(self):
        turn_acc_score = self.slot_turn_acc / self.all_hit
        slot_F1_score = self.slot_F1_pred / self.slot_F1_count
        joint_goal_accuracy = self.joint_goal_hit / self.all_hit
        eval_result = {
            "joint_goal_accuracy": joint_goal_accuracy,
            "turn_slot_accuracy": turn_acc_score,
            "turn_slot_f1": slot_F1_score,
        }
        return eval_result


def compute_acc(gold, pred, slot_meta):
    miss_gold = 0
    miss_slot = []
    miss_labels = []  # value까지 체크하기 위해 추가
    for g in gold:
        if g not in pred:
            miss_gold += 1  # 정확히 예측 못한 것
            miss_slot.append(g.rsplit("-", 1)[0])
            
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1  # 예측하지 말아야 할 것을 예측한 것
    
    # 틀린 예측 slot을 체크하기 위해 저장하는 list 생성
    for slot in miss_slot:
        gold_values = [s.rsplit('-', 1)[1] for s in gold if slot in s] 
        miss_values = [s.rsplit('-', 1)[1] for s in pred if slot in s] 
        for label, pred in zip(gold_values, miss_values):
            miss_labels.append({'slot': slot, 'label':label, 'pred': pred})

    ACC_TOTAL = len(slot_meta)
    ACC = len(slot_meta) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC, miss_labels


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count
