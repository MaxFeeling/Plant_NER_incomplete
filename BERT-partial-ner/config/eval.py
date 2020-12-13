import numpy as np
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

def predict_json(batch_insts: List[Instance],
                 batch_pred_ids: torch.Tensor,
                 batch_gold_ids: torch.LongTensor,
                 word_seq_lens: torch.LongTensor,
                 idx2label: List[str]) -> np.ndarray:
    word_seq_lens = word_seq_lens.tolist()
    test_submit = []
    if batch_gold_ids is not None:
        gold_submit = []
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]

        #predict labels
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        prediction = [idx2label[l] for l in prediction]
        json_d = {}
        json_d["id"] = batch_insts[idx].id
        json_d["query"] = "".join(batch_insts[idx].input.words)
        start = -1
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
                # if i == len(prediction) - 1 or prediction[i + 1].startswith("B-") or prediction[i + 1].startswith("O"):
                #     type = prediction[i][2:]
                #     entity = {}
                #     entity["str"] = "".join(batch_insts[idx].input.words[start])
                #     entity["start_position"] = i
                #     entity["end_position"] = i
                #
                #     if type not in json_d:
                #         json_d[type] = []
                #     json_d[type].append(entity)
            if prediction[i].startswith("E-") and start != -1:
                end = i
                type = prediction[i][2:]
                entity = {}
                entity["str"] = "".join(batch_insts[idx].input.words[start:end + 1])
                entity["start_position"] = start
                entity["end_position"] = end

                if type not in json_d:
                    json_d[type] = []
                json_d[type].append(entity)

        # gold_labels
        if batch_gold_ids is not None:
            output = batch_gold_ids[idx][:length].tolist()
            output = [idx2label[l] for l in output]
            json_o = {}
            json_o["id"] = batch_insts[idx].id
            json_o["query"] = "".join(batch_insts[idx].input.words)
            start = -1
            for i in range(len(output)):
                if output[i].startswith("B-"):
                    start = i
                    if i == len(output) - 1 or output[i + 1].startswith("B-") or output[i + 1].startswith("O"):
                        type = output[i][2:]
                        entity = {}
                        entity["str"] = "".join(batch_insts[idx].input.words[start])
                        entity["start_position"] = i
                        entity["end_position"] = i

                        if type not in json_o:
                            json_o[type] = []
                        json_o[type].append(entity)
                if output[i].startswith("E-") and start != -1:
                    end = i
                    type = output[i][2:]
                    entity = {}
                    entity["str"] = "".join(batch_insts[idx].input.words[start:end + 1])
                    entity["start_position"] = start
                    entity["end_position"] = end
                    if type not in json_o:
                        json_o[type] = []
                    json_o[type].append(entity)
        # gold_submit.append(json_o)
        test_submit.append(json_d)
        # json_to_text(submit_file,test_submit)
    if batch_gold_ids is None:
        return test_submit
    return test_submit, gold_submit

def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.Tensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str]) -> np.ndarray:
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
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        # convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
                if i == len(output) - 1 or output[i + 1].startswith("B-") or output[i + 1].startswith("O"):
                    output_spans.add(Span(i, i, output[i][2:]))
            if output[i].startswith("E-") and start != -1:
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
                # if i == len(prediction) - 1 or prediction[i + 1].startswith("B-") or prediction[i + 1].startswith("O"):
                #     predict_spans.add(Span(i, i, prediction[i][2:]))
            if prediction[i].startswith("E-") and start!=-1:
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))

    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return np.asarray([p, total_predict, total_entity], dtype=int)
