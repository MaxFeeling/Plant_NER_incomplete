import argparse
import random
import numpy as np
from typing import Tuple
from config import Reader, Config, evaluate_batch_insts, batching_list_instances,predict_json
import time
import torch
from typing import List
from common import Instance
import os
import logging
import pickle
import math
import itertools
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from bert_model import BertCRF
import utils
from bert_softmax import BertSoftMax
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments_t(parser):
    # Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda", choices=['cpu', 'cuda'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=2019, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="data")
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help="default batch size is 32 (works well)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Usually we set to 10.")  # origin 100
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--num_outer_iterations', type=int, default=20, help="Number of outer iterations for cross validation")

    # bert hyperparameter
    parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch', help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--max_len', default=180, help="max allowed sequence length")
    parser.add_argument('--full_finetuning', default=True, action='store_true',
                        help="Whether to fine tune the pre-trained model")
    parser.add_argument('--clip_grad', default=5, help="gradient clipping")
    # model hyperparameter
    parser.add_argument('--model_folder', type=str, default="saved_model", help="The name to save the model files")
    parser.add_argument('--device_num', type=str, default='0,1', help="The gpu number you want to use")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, train_insts: List[List[Instance]], dev_insts: List[Instance]):
    train_num = sum([len(insts) for insts in train_insts])
    logging.info(("[Training Info] number of instances: %d" % (train_num)))
    # get the batched data
    dev_batches = batching_list_instances(config, dev_insts)

    model_folder = config.model_folder

    logging.info("[Training Info] The model will be saved to: %s" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    num_outer_iterations = config.num_outer_iterations

    for iter in range(num_outer_iterations):

        logging.info(f"[Training Info] Running for {iter}th large iterations.")

        model_names = []  # model names for each fold

        train_batches = [batching_list_instances(config, insts) for insts in train_insts]

        logging.info("length of train_insts：%d"% len(train_insts))

        # train 2 models in 2 folds
        for fold_id, folded_train_insts in enumerate(train_insts):
            logging.info(f"[Training Info] Training fold {fold_id}.")
            # Initialize bert model
            logging.info("Initialized from pre-trained Model")

            crf_model_name = model_folder + f"/bert_crf_{fold_id}"
            msra_model_name = model_folder+ f"/bert_msra_{fold_id}"
            model_names.append(crf_model_name)
            model_names.append(msra_model_name)
            # if os.path.exists(crf_model_name) and os.path.exists(msra_model_name):
            #     pass
            # else:
            # torch.cuda.empty_cache()
            train_one(config=config, train_batches=train_batches[fold_id],
                      dev_insts=dev_insts, dev_batches=dev_batches, crf_model_name=crf_model_name,msra_model_name=msra_model_name)

        # assign prediction to other folds
        logging.info("\n\n")
        logging.info("[Data Info] Assigning labels")

        # using the model trained in one fold to predict the result of another fold's data
        # and update the label of another fold with the predict result
        for fold_id, folded_train_insts in enumerate(train_insts):

            cfig_path = os.path.join(config.bert_model_dir, 'bert_config.json')
            cfig = BertConfig.from_json_file(cfig_path)
            cfig.device = config.device
            cfig.label2idx = config.label2idx
            cfig.label_size = config.label_size
            cfig.idx2labels = config.idx2labels

            msra_model_name = model_folder + f"/bert_msra_{fold_id}"
            crf_model_name = model_folder + f"/bert_crf_{fold_id}"

            msra_model = BertSoftMax(cfig=cfig)
            msra_model.to(cfig.device)
            # msra_model = nn.DataParallel(msra_model,device_ids=[0,1])
            utils.load_checkpoint(os.path.join(msra_model_name, 'best.pth.tar'), msra_model)


            crf_model = BertCRF(cfig=cfig)
            crf_model.to(cfig.device)
            # crf_model = nn.DataParallel(crf_model, device_ids=[0, 1])
            utils.load_checkpoint(os.path.join(crf_model_name, 'best.pth.tar'), crf_model)


            hard_constraint_predict(config=config, msra_model=msra_model,crf_model=crf_model,
                                    fold_batches=train_batches[1 - fold_id],
                                    folded_insts=train_insts[1 - fold_id])  # set a new label id, k is set to 2, so 1 - fold_id can be used
        logging.info("\n\n")

        logging.info("[Training Info] Training the final model")

        # merge the result data to training the final model
        all_train_insts = list(itertools.chain.from_iterable(train_insts))

        logging.info("Initialized from pre-trained Model")

        # msra_model_name = model_folder + "/final_bert_msra"
        final_model_name = model_folder + "/final_bert_crf_1"
        config_name = model_folder + "/config.conf"

        all_train_batches = batching_list_instances(config=config, insts=all_train_insts)
        # train the final model
        final_model = train_one(config=config, train_batches=all_train_batches, dev_insts=dev_insts, dev_batches=dev_batches,
                          crf_model_name=final_model_name,msra_model_name="final", config_name=config_name)
        # load the best final model
        # utils.load_checkpoint(os.path.join(msra_model_name, 'best.pth.tar'), msra_model)
        if isinstance(final_model, nn.DataParallel):
            final_model = final_model.module
        # if isinstance(msra_model, nn.DataParallel):
        #     msra_model = msra_model.module
        # utils.load_checkpoint(os.path.join(msra_model_name, 'best.pth.tar'), msra_model)
        utils.load_checkpoint(os.path.join(final_model_name, 'best.pth.tar'), final_model)

        # msra_model.eval()
        final_model.eval()
        logging.info("\n")
        result = evaluate_model(config, crf_model=final_model,msra_model=None, batch_insts_ids=dev_batches, name="dev", insts=dev_insts)
        logging.info("\n\n")


def hard_constraint_predict(config: Config, crf_model: BertCRF, msra_model:BertSoftMax, fold_batches: List[Tuple], folded_insts:List[Instance], model_type:str = "hard"):
    """using the model trained in one fold to predict the result of another fold"""
    batch_id = 0
    batch_size = config.batch_size
    if isinstance(crf_model, nn.DataParallel):
        crf_model = crf_model.module
    if isinstance(msra_model, nn.DataParallel):
        msra_model = msra_model.module
    msra_model.eval()
    crf_model.eval()

    with torch.no_grad():
        for batch in fold_batches:
            one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]

            input_ids, input_seq_lens, annotation_mask, labels = batch
            input_masks = input_ids.gt(0)
            # get the predict result

            outputs = msra_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=None, labels=None,attention_mask=input_masks)
            label_prob = outputs[0]
            label_prob = torch.nn.functional.softmax(label_prob, dim=2)
            batch_max_scores, batch_max_ids = crf_model(input_ids, input_seq_lens=input_seq_lens,
                                                        annotation_mask=annotation_mask,
                                                        labels=None, attention_mask=input_masks, label_prob=label_prob)
            # except RuntimeError as exception:
            #     if "out of memory" in str(exception):
            #         print("WARNING: out of memory")
            #         if hasattr(torch.cuda, 'empty_cache'):
            #             torch.cuda.empty_cache()
            #     else:
            #         raise exception
            # batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens,
            #                                         annotation_mask=annotation_mask, labels=None, attention_mask=input_masks)

            batch_max_ids = batch_max_ids.cpu().numpy()
            word_seq_lens = batch[1].cpu().numpy()
            for idx in range(len(batch_max_ids)):
                length = word_seq_lens[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1]
                # update the labels of another fold
                one_batch_insts[idx].output_ids = prediction
            batch_id += 1

def predict(output_file:str,gold_file:str):
    logging.info("Predict Implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset+"/train.txt"
    conf.test_file = conf.dataset + "/TestData.json"

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num
    # data reader
    reader = Reader(conf.digit2zero)
    reader_no = Reader(False)
    set_seed(opt, conf.seed)

    logging.info("Loading the datasets...")
    train_insts = reader.read_txt(conf.train_file,conf.train_num)
    test_insts = reader.read_json(conf.test_file, conf.test_num)
    test_no =reader_no.read_json(conf.test_file, conf.test_num)
    logging.info("Building label idx ...")
    # build label2idx and idx2label
    conf.build_label_idx(train_insts+test_insts)
    test_batches =batching_list_instances(conf,test_insts)

    logging.info("Model implementation...")
    cfig_path = os.path.join(conf.bert_model_dir,
                             'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = conf.device
    cfig.label2idx = conf.label2idx
    cfig.label_size = conf.label_size
    cfig.idx2labels = conf.idx2labels
    # load pretrained bert model
    crf_model = BertCRF.from_pretrained(conf.bert_model_dir, config=cfig)
    crf_model.to(conf.device)
    utils.load_checkpoint(checkpoint="saved_model/final_bert_crf_fintune/71.35_2gpu.pth (3).tar",model=crf_model)

    logging.info("Predict...")
    crf_model.eval()
    batch_id =0
    batch_size = conf.batch_size
    json_submit = []
    # gold_submit = []
    for batch in test_batches:
        one_batch_insts = test_no[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result

        batch_max_scores, batch_max_ids = crf_model(input_ids, input_seq_lens=input_seq_lens,
                                                    annotation_mask=annotation_mask,
                                                    labels=None, attention_mask=input_masks, label_prob=None)
        json = predict_json(batch_insts=one_batch_insts,
                                 batch_pred_ids=batch_max_ids,
                                 batch_gold_ids=None,
                                 word_seq_lens=batch[1], idx2label=conf.idx2labels)
        json_submit+=json
        # gold_submit+=gold
        batch_id+=1
    utils.json_to_text(output_file,json_submit)
    # utils.json_to_text(gold_file,gold_submit)

def train_one(config: Config, train_batches: List[Tuple], dev_insts: List[Instance], dev_batches: List[Tuple],
              crf_model_name: str,msra_model_name:str, config_name: str = None) -> BertCRF:

    # load config for bertCRF
    cfig_path = os.path.join(config.bert_model_dir,
                             'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = config.device
    cfig.label2idx = config.label2idx
    cfig.label_size = config.label_size
    cfig.idx2labels = config.idx2labels
    # load pretrained bert model
    if msra_model_name!="final":
        crf_model = BertCRF.from_pretrained(config.bert_model_dir, config=cfig)
        crf_model.to(config.device)

        msra_model = BertSoftMax.from_pretrained(config.bert_model_dir,config=cfig)
        msra_model.to(config.device)



        if config.full_finetuning:
            logging.info('full finetuning')
            crf_param_optimizer = list(crf_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            crf_grouped_parameters = [
                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

            msra_param_optimizer = list(msra_model.named_parameters())
            msra_grouped_parameters = [
                {'params': [p for n, p in msra_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in msra_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

            # msra_param_optimizer = list(msra_model.classifier.named_parameters())
            # msra_grouped_parameters = [{'params': [p for n, p in msra_param_optimizer]}]

        else:
            logging.info('tuning downstream layer')
            crf_param_optimizer = list(crf_model.classifier.named_parameters())
            crf_grouped_parameters = [{'params': [p for n, p in crf_param_optimizer]}]
            msra_param_optimizer = list(msra_model.classifier.named_parameters())
            msra_grouped_parameters = [{'params': [p for n, p in msra_param_optimizer]}]


        optimizer_grouped_parameters = crf_grouped_parameters+msra_grouped_parameters
        optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        msra_model = torch.nn.DataParallel(msra_model,device_ids=[0,1])
        crf_model = torch.nn.DataParallel(crf_model,device_ids=[0,1])
        utils.load_checkpoint(checkpoint="saved_model/bert_pre_train/ne_best.pth (2).tar", model=msra_model)
        crf_model.train()
        msra_model.train()

        epoch = config.num_epochs
        best_dev_f1 = -1
        for i in range(1, epoch + 1):
            epoch_loss = 0
            start_time = time.time()
            crf_model.zero_grad()
            msra_model.zero_grad()

            for index in np.random.permutation(len(train_batches)):  # disorder the train batches
                crf_model.train()
                msra_model.train()
                scheduler.step()
                input_ids, input_seq_lens, annotation_mask, labels = train_batches[index]
                input_masks = input_ids.gt(0)
                # update loss
                outputs = msra_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                                     labels=None, attention_mask=input_masks)
                logits = outputs[0]
                label_prob = torch.nn.functional.softmax(logits, dim=2)
                loss = crf_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                                     labels=labels, attention_mask=input_masks, label_prob=label_prob)

                loss = loss.mean()
                epoch_loss += loss.item()
                crf_model.zero_grad()
                msra_model.zero_grad()
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(parameters=msra_model.parameters(), max_norm=config.clip_grad)
                nn.utils.clip_grad_norm_(parameters=crf_model.parameters(), max_norm=config.clip_grad)
                optimizer.step()
            end_time = time.time()
            logging.info("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))
            # torch.cuda.empty_cache()
            msra_model.eval()
            crf_model.eval()
            with torch.no_grad():
                # metric is [precision, recall, f_score]
                dev_metrics = evaluate_model(config, crf_model,msra_model, dev_batches, "dev", dev_insts)
                if dev_metrics[2] > best_dev_f1:  # save the best model
                    logging.info("saving the best model...")
                    best_dev_f1 = dev_metrics[2]

                    crf_model_to_save = crf_model.module if hasattr(crf_model, 'module') else crf_model  # Only save the model it-self
                    optimizer_to_save = optimizer
                    utils.save_checkpoint({'epoch': epoch + 1,

                                           'state_dict': crf_model_to_save.state_dict(),
                                           'optim_dict': optimizer_to_save.state_dict()},
                                          is_best=dev_metrics[2] > 0,
                                          checkpoint=crf_model_name)

                    msra_model_to_save = msra_model.module if hasattr(msra_model,'module') else msra_model  # Only save the model it-self
                    # optimizer_to_save = optimizer
                    utils.save_checkpoint({'epoch': epoch + 1,
                                           'gpu_nums': 2,
                                           'state_dict': msra_model_to_save.state_dict()},
                                          is_best=dev_metrics[2] > 0,
                                          checkpoint=msra_model_name)
                    # Save the corresponding config as well.
                    if config_name:
                        f = open(config_name, 'wb')
                        pickle.dump(config, f)
                        f.close()
            msra_model.zero_grad()
            crf_model.zero_grad()
            # torch.cuda.empty_cache()

    else:
        crf_model = BertCRF.from_pretrained(config.bert_model_dir, config=cfig)

        crf_model.to(config.device)


        if config.full_finetuning:
            logging.info('full finetuning')
            crf_param_optimizer = list(crf_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            crf_grouped_parameters = [
                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

        else:
            logging.info('tuning downstream layer')
            crf_param_optimizer = list(crf_model.classifier.named_parameters())
            crf_grouped_parameters = [{'params': [p for n, p in crf_param_optimizer]}]
        optimizer = Adam(crf_grouped_parameters, lr=config.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        crf_model = torch.nn.DataParallel(crf_model,device_ids=[0,1])
        crf_model.train()


        epoch = config.num_epochs
        best_dev_f1 = -1
        for i in range(1, epoch + 1):
            epoch_loss = 0
            start_time = time.time()
            crf_model.zero_grad()

            for index in np.random.permutation(len(train_batches)):  # disorder the train batches
                crf_model.train()
                scheduler.step()
                input_ids, input_seq_lens, annotation_mask, labels = train_batches[index]
                input_masks = input_ids.gt(0)
                # update loss
                loss = crf_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                                 labels=labels, attention_mask=input_masks,label_prob=None)
                loss=loss.mean()
                epoch_loss += loss.item()
                crf_model.zero_grad()

                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(parameters=crf_model.parameters(), max_norm=config.clip_grad)
                optimizer.step()
            end_time = time.time()
            logging.info("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))
            torch.cuda.empty_cache()
            crf_model.eval()
            with torch.no_grad():
                # metric is [precision, recall, f_score]
                dev_metrics = evaluate_model(config, crf_model, msra_model=None, batch_insts_ids=dev_batches,name="dev", insts=dev_insts)
                if dev_metrics[2] > best_dev_f1:  # save the best model
                    logging.info("saving the best model...")
                    best_dev_f1 = dev_metrics[2]

                    crf_model_to_save = crf_model.module if hasattr(crf_model,
                                                                    'module') else crf_model  # Only save the model it-self
                    optimizer_to_save = optimizer
                    utils.save_checkpoint({'epoch': epoch + 1,
                                           'gpu_nums': 2,
                                           'state_dict': crf_model_to_save.state_dict(),
                                           'optim_dict': optimizer_to_save.state_dict()},
                                          is_best=dev_metrics[2] > 0,
                                          checkpoint=crf_model_name)

                    # Save the corresponding config as well.
                    if config_name:
                        f = open(config_name, 'wb')
                        pickle.dump(config, f)
                        f.close()
            crf_model.zero_grad()
            torch.cuda.empty_cache()
    if msra_model_name!="final":
        return msra_model, crf_model
    return crf_model
def evaluate_model(config: Config, crf_model: BertCRF,msra_model:BertSoftMax, batch_insts_ids, name: str, insts: List[Instance]):
    # evaluation
    metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result
        if msra_model is not None:
            if isinstance(msra_model, nn.DataParallel):
                msra_model = msra_model.module
            if isinstance(crf_model, nn.DataParallel):
                crf_model = crf_model.module
            outputs = msra_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=None, labels=None,
                              attention_mask=input_masks)
            label_prob = outputs[0]
            label_prob = torch.nn.functional.softmax(label_prob, dim=2)
            batch_max_scores, batch_max_ids = crf_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                             labels=None, attention_mask=input_masks,label_prob=label_prob)
        else:
            if isinstance(crf_model, nn.DataParallel):
                crf_model = crf_model.module
            batch_max_scores, batch_max_ids = crf_model(input_ids, input_seq_lens=input_seq_lens,
                                                        annotation_mask=annotation_mask,
                                                        labels=None, attention_mask=input_masks, label_prob=None)

        metrics += evaluate_batch_insts(batch_insts=one_batch_insts,
                                        batch_pred_ids=batch_max_ids,
                                        batch_gold_ids=batch[-1],
                                        word_seq_lens=batch[1], idx2label=config.idx2labels)
        batch_id += 1
    # calculate the precision, recall and f1 score
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    logging.info("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore))
    return [precision, recall, fscore]


def main():
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train.txt"
    conf.dev_file = conf.dataset + "/valid.txt"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num
    # data reader
    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    # set logger
    utils.set_logger(os.path.join(conf.model_folder, 'train.log'))

    # params
    for k in opt.__dict__:
        logging.info(k + ": " + str(opt.__dict__[k]))

    # read trains/devs
    logging.info("\n")
    logging.info("Loading the datasets...")
    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)

    logging.info("Building label idx ...")
    # build label2idx and idx2label
    conf.build_label_idx(trains + devs)

    random.shuffle(trains)
    # set the prediction flag, if is_prediction is False, we will not update this label.
    for inst in trains:
        inst.is_prediction = [False] * len(inst.input)
        for pos, label in enumerate(inst.output):
            if label == conf.O:
                inst.is_prediction[pos] = True
    # dividing the data into 2 parts(num_folds default to 2)
    num_insts_in_fold = math.ceil(len(trains) / conf.num_folds)
    trains = [trains[i * num_insts_in_fold: (i + 1) * num_insts_in_fold] for i in range(conf.num_folds)]

    train_model(config=conf, train_insts=trains, dev_insts=devs)


if __name__ == "__main__":
    main()
    # predict('YourTeamName.json','ValidLabel.json')

    # final_model_name = model_folder + "/final_bert_crf_test"
    # utils.load_checkpoint(checkpoint="saved_model/final_bert_crf_5e-5/best.pth.tar", model=crf_model)
    # if conf.full_finetuning:
    #     logging.info('full finetuning')
    #     crf_param_optimizer = list(crf_model.named_parameters())
    #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #     crf_grouped_parameters = [
    #         {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.01},
    #         {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.0}
    #     ]
    #
    # else:
    #     logging.info('tuning downstream layer')
    #     crf_param_optimizer = list(crf_model.classifier.named_parameters())
    #     crf_grouped_parameters = [{'params': [p for n, p in crf_param_optimizer]}]
    # optimizer = Adam(crf_grouped_parameters, lr=conf.learning_rate)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
    #
    # crf_model.train()
    #
    # epoch = conf.num_epochs
    # best_dev_f1 = -1
    # for i in range(1, epoch + 1):
    #     epoch_loss = 0
    #     start_time = time.time()
    #     crf_model.zero_grad()
    #
    #     for index in np.random.permutation(len(test_batches)):  # disorder the train batches
    #         crf_model.train()
    #         scheduler.step()
    #         input_ids, input_seq_lens, annotation_mask, labels = test_batches[index]
    #         input_masks = input_ids.gt(0)
    #         # update loss
    #         loss = crf_model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
    #                          labels=labels, attention_mask=input_masks, label_prob=None)
    #
    #         epoch_loss += loss.item()
    #         crf_model.zero_grad()
    #
    #         loss.backward()
    #         # gradient clipping
    #         nn.utils.clip_grad_norm_(parameters=crf_model.parameters(), max_norm=conf.clip_grad)
    #         optimizer.step()
    #     end_time = time.time()
    #     logging.info("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))
    #     print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))
    #     torch.cuda.empty_cache()
    #     crf_model.eval()
    #     with torch.no_grad():
    #         # metric is [precision, recall, f_score]
    #         dev_metrics = evaluate_model(conf, crf_model, msra_model=None, batch_insts_ids=test_batches, name="dev",
    #                                      insts=test_no)
    #         print(dev_metrics)
    #         if dev_metrics[2] > best_dev_f1:  # save the best model
    #             logging.info("saving the best model...")
    #             best_dev_f1 = dev_metrics[2]
    #
    #             crf_model_to_save = crf_model.module if hasattr(crf_model,
    #                                                             'module') else crf_model  # Only save the model it-self
    #             optimizer_to_save = optimizer
    #             utils.save_checkpoint({'epoch': epoch + 1,
    #
    #                                    'state_dict': crf_model_to_save.state_dict(),
    #                                    'optim_dict': optimizer_to_save.state_dict()},
    #                                   is_best=dev_metrics[2] > 0,
    #                                   checkpoint=final_model_name)
    # crf_model.eval()
    # result=evaluate_model(conf,crf_model=crf_model,msra_model=None,batch_insts_ids=test_batches,name="dev",insts=test_no)
    # print(result)