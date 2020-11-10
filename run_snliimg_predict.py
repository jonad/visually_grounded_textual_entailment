# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
from run_sembert_img_classifier import *
from run_classifier import *
import numpy as np
import torch
from torchvision import transforms

from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from tag_model.modeling import TagConfig
from data_process.datasets import SenSequence, DocSequence, QuerySequence, QueryTagSequence, \
    DocTagSequence
from pytorch_pretrained_bert.modeling import BertForSequenceClassificationTag
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tag_model.tag_tokenization import TagTokenizer
from tag_model.tagging import get_tags, SRLPredictor
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
label_map = {}
#csv.field_size_limit(sys.maxsize)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="glue_data/MNLI/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="mnli",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="base_mnli",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tagger_path", default=None, type=str,
                        help="tagger_path for predictions if needing real-time tagging. Default: None, by loading pre-tagged data"
                             "For example, the trained models by AllenNLP")
    parser.add_argument("--max_num_aspect",
                        default=3,
                        type=int,
                        help="max_num_aspect")
    parser.add_argument("--model_filename",
                        default="pytorch_model.bin",
                        type=str,
                        help="model filename")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "snliimg": SnliImgProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    label_map = processor.get_labels_map()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.tagger_path != None:
        srl_predictor = SRLPredictor(args.tagger_path)
    else:
        srl_predictor = None

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    tag_tokenizer = TagTokenizer()
    vocab_size = len(tag_tokenizer.ids_to_tags)
    print("tokenizer vocab size: ", str(vocab_size))
    tag_config = TagConfig(tag_vocab_size=vocab_size,
                           hidden_size=10,
                           layer_num=1,
                           output_dim=10,
                           dropout_prob=0.1,
                           num_aspect=args.max_num_aspect)

    # Prepare optimizer

    if args.do_eval:
        # for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, srl_predictor=srl_predictor)
        eval_features = transform_tag_features(args.max_num_aspect, eval_features, tag_tokenizer,
                                                   args.max_seq_length)
            
        all_input_ids = [f.input_ids for f in eval_features]
        all_input_mask = [f.input_mask for f in eval_features]
        all_segment_ids = [f.segment_ids for f in eval_features]
        all_label_ids = [f.label_id for f in eval_features]
        all_start_end_idx = [f.orig_to_token_split_idx for f in eval_features]
        all_input_tag_ids = [f.input_tag_ids for f in eval_features]
        all_images = [f.image for f in eval_features]
            
        eval_data = SequenceImageDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx,
                                             all_input_tag_ids,
                                             all_label_ids, all_images, transform, IMAGE_DIR)
            
        logger.info("***** Evaluation data *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # epoch = 1
        output_model_file = os.path.join(args.output_dir,args.model_filename)
        model_state_dict = torch.load(output_model_file)
        predict_model = model = BertForSequenceImgClassificationTag.from_pretrained(args.bert_model,
                                                                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                        args.local_rank),
                                                                    num_labels=num_labels, tag_config=tag_config,
                                                                    image_emb_size=2048)
        predict_model.to(device)
        predict_model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        total_precision = np.zeros(3)
        total_recall = np.zeros(3)
        total_fscore = np.zeros(3)
        total_support = np.zeros(3, dtype=int)
      


        for batch_number, data_value in enumerate(tqdm(
                eval_dataloader, desc="Evaluating")):
            input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, images, label_ids = data_value
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            start_end_idx = start_end_idx.to(device)
            input_tag_ids = input_tag_ids.to(device)
            images = images.to(device)
            with torch.no_grad():
                tmp_eval_loss = predict_model(input_ids, segment_ids, input_mask, start_end_idx,
                                                      input_tag_ids, images, label_ids)
                logits = predict_model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids,
                                               images, None)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            tmp_eval_accuracy = accuracy(logits, label_ids)
            precision, recall, fscore, support = precision_recall_fscore_support(label_ids, np.argmax(logits, axis=1), labels=[0, 1, 2])
            total_precision = total_precision + precision
            total_recall = total_recall + recall
            total_fscore = total_fscore + fscore
            total_support = total_support + support


            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        total_precision = total_precision / (batch_number + 1)
        total_recall = total_recall / (batch_number + 1)
        total_fscore = total_fscore/ (batch_number + 1)

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'total_precision':{
                    k: total_precision.tolist()[v] for k, v in label_map.items()
                  },
                  'total_recall': {
                    k: total_recall.tolist()[v] for k, v in label_map.items()
                  },
                  'total_fscore': {
                    k: total_fscore.tolist()[v] for k, v in label_map.items()
                  },
                  'total_support':{
                    k: total_support.tolist()[v] for k, v in label_map.items()
                  },
                  'macro_precision': total_precision.mean(),
                  'macro_recall': total_recall.mean(),
                  'macro_support': total_support.sum(),
                  'number_of_examples': nb_eval_examples }
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("Result:  %s = %s",  key, str(result[key]))
                writer.write("Result: %s = %s\n" % (key, str(result[key])))
        logger.info("result:  %s", str(result))

    if args.do_predict:
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer,srl_predictor=srl_predictor )
        eval_features = transform_tag_features(args.max_num_aspect, eval_features, tag_tokenizer, args.max_seq_length)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
        all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx,
                                  all_input_tag_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        predict_model = BertForSequenceClassificationTag.from_pretrained(args.bert_model, state_dict=model_state_dict,num_labels = num_labels,tag_config=tag_config)
        predict_model.to(device)
        predict_model.eval()
        predictions = []

        for input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_end_idx = start_end_idx.to(device)
            input_tag_ids = input_tag_ids.to(device)
            with torch.no_grad():
                logits = predict_model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None)
            logits = logits.detach().cpu().numpy()
            for (i, prediction) in enumerate(logits):
                predict_label = np.argmax(prediction)
                predictions.append(predict_label)

        output_test_file = os.path.join(args.output_dir, "_pred_results.tsv")
        index = 0
        with open(output_test_file, "w") as writer:
            writer.write("index" + "\t" + "prediction" + "\n")
            for pred in predictions:
                writer.write(str(index) + "\t" + str(label_list[int(pred)]) + "\n")
                index += 1

if __name__ == "__main__":
    main()
