
import numpy as np
from overrides import overrides
from typing import List
from common import Instance
import torch


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))


def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str],
                         use_crf_layer: bool = True,
                         kind: str = "MISC",
                         flag: int = 0) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    p = 0
    p_special = 0
    total_entity = 0
    total_predict = 0
    special_entity = 0
    special_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    list_output_spans = []
    list_predict_spans = []
    wrong_prediction = {}
    wrong_prediction["BLater"] = 0
    wrong_prediction["BEarlier"] = 0
    wrong_prediction["ILater"] = 0
    wrong_prediction["IEarlier"] = 0
    wrong_prediction["O2misc"] = 0
    wrong_prediction["misc2O"] = 0
    wrong_prediction[1] = 0
    wrong_prediction[2] = 0
    wrong_prediction[3] = 0
    wrong_prediction[4] = 0
    wrong_prediction[5] = 0
    wrong_prediction[6] = 0
    wrong_prediction[7] = 0
    wrong_prediction["length1"] = 0
    wrong_prediction["length2"] = 0
    wrong_prediction["length3"] = 0
    wrong_prediction["length4"] = 0
    wrong_prediction["length5"] = 0
    wrong_prediction["length6"] = 0
    wrong_prediction["length7"] = 0
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1] if use_crf_layer else prediction
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        #convert to span
        output_spans = set()
        output_special_spans = set()
        start = -1
        start_special = -1
        # end_special = 0
        # for i in range(len(output)):
        #     print(output[i], prediction[i])
        # for i in range(len(output) - 1):
        #     if output[i] == "B-misc" and output[i + 1] == "I-misc" and prediction[i] == "O-misc" and prediction[i + 1] == "B-misc":
        #         wrong_prediction["BLater"] += 1
        #     elif output[i + 1] == "B-misc" and prediction[i] == "B-misc":
        #         wrong_prediction["BEarlier"] += 1
        #     elif output[i + 1] == "E-misc" and prediction[i] == "E-misc":
        #         wrong_prediction["IEarlier"] += 1
        #     elif output[i] == "E-misc" and prediction[i+1] == "E-misc":
        #         wrong_prediction["ILater"] += 1
        #     elif ~(output[i].endswith("-misc")) and prediction[i].endswith("-misc"):
        #         wrong_prediction["O2misc"] += 1
        #     elif ~(prediction[i].endswith("-misc")) and output[i].endswith("-misc"):
        #         wrong_prediction["misc2O"] += 1
            
        
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].endswith(kind) and output[i].startswith("B-"):
                start_special = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
                list_output_spans.append(Span(start, end, output[i][2:]))
            if output[i].endswith(kind) and output[i].startswith("E-"):
                end_special = i
                output_special_spans.add(Span(start_special, end_special, output[i][2:]))
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
            if output[i].endswith(kind) and output[i].startswith("S-"):
                output_special_spans.add(Span(i, i, output[i][2:]))
        
        predict_spans = set()
        predict_special_spans = set()
        start = -1
        start_special = -1
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].endswith(kind) and prediction[i].startswith("B-"):
                start_special = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
                list_predict_spans.append(Span(start, end, output[i][2:]))
            if prediction[i].endswith(kind) and prediction[i].startswith("E-"):
                end_special = i
                predict_special_spans.add(Span(start_special, end_special, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
            if prediction[i].endswith(kind) and prediction[i].startswith("S-"):
                predict_special_spans.add(Span(i, i, prediction[i][2:]))

        total_entity += len(output_spans)
        special_entity += len(output_special_spans)
                
        total_predict += len(predict_spans)
        special_predict += len(predict_special_spans)

        p += len(predict_spans.intersection(output_spans))
        p_special += len(predict_spans.intersection(output_special_spans))

    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0


        for inn in output_special_spans:
            flag1 = 0
            for jnn in predict_spans:
                if inn.left == jnn.left and inn.right == jnn.right and jnn.type == kind:
                    flag1 = 1
                    break
            if flag1 == 0:
                for jnn in predict_spans:
                    if inn.left >= jnn.left and inn.right <= jnn.right:
                        print()
                        for ran in range(inn.left, inn.right + 1):
                            print((batch_insts[idx].input.words)[ran], end=" ")
                        print()
                        print(inn.left, inn.right, inn.type)
                        print(jnn.left, jnn.right, jnn.type)
                        print()
            if flag1 == 0 and inn.right - inn.left + 1 <= 7:
                wrong_prediction["length" + str(inn.right - inn.left + 1)] += 1

        for inn in output_spans:
            for jnn in predict_spans:
                if inn.left > jnn.left and inn.left <= jnn.right and inn.type == jnn.type and inn.type == kind:
                    wrong_prediction["BEarlier"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.left < jnn.left and inn.right >= jnn.left and inn.type == jnn.type and inn.type == kind:
                    wrong_prediction["BLater"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.right > jnn.right and inn.left <= jnn.right and inn.type == jnn.type and inn.type == kind:
                    wrong_prediction["IEarlier"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.right < jnn.right and inn.right >= jnn.left and inn.type == jnn.type and inn.type == kind:
                    wrong_prediction["ILater"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.left <= jnn.left and inn.right >= jnn.left and inn.type != kind and jnn.type == kind:
                    wrong_prediction["O2misc"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.left >= jnn.left and inn.left <= jnn.right and inn.type != kind and jnn.type == kind:
                    wrong_prediction["O2misc"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.left == inn.right and inn.left == jnn.left and jnn.left == jnn.right and inn.type != kind and jnn.type == kind:
                    wrong_prediction["O2misc"] -= 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] -= 1
                if inn.left <= jnn.left and inn.right >= jnn.left and inn.type == kind and jnn.type != kind:
                    wrong_prediction["misc2O"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.left >= jnn.left and inn.left <= jnn.right and inn.type == kind and jnn.type != kind:
                    wrong_prediction["misc2O"] += 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] += 1
                if inn.left == inn.right and inn.left == jnn.left and jnn.left == jnn.right and inn.type == kind and jnn.type != kind:
                    wrong_prediction["misc2O"] -= 1
                    if inn.right - inn.left + 1 <= 7:
                        wrong_prediction[inn.right - inn.left + 1] -= 1
    # print(wrong_prediction)        
    
    return np.asarray([p, p_special, total_predict, total_entity, special_predict, special_entity, 
    wrong_prediction["BLater"], 
    wrong_prediction["BEarlier"],
    wrong_prediction["ILater"],
    wrong_prediction["IEarlier"],
    wrong_prediction["O2misc"],
    wrong_prediction["misc2O"],
    wrong_prediction[1],
    wrong_prediction[2],
    wrong_prediction[3],
    wrong_prediction[4],
    wrong_prediction[5],
    wrong_prediction[6],
    wrong_prediction[7],
    wrong_prediction["length1"],
    wrong_prediction["length2"],
    wrong_prediction["length3"],
    wrong_prediction["length4"],
    wrong_prediction["length5"],
    wrong_prediction["length6"],
    wrong_prediction["length7"]], dtype=int)
