# coding=utf-8

from run_classifier import Dataset, DataProcessor, InputFeatures, \
    InputExample, convert_examples_to_features, transform_tag_features, accuracy
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
import os
from PIL import Image
import torch
import argparse
import logging
import random
import numpy as np
from tag_model.tagging import get_tags, SRLPredictor
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from tag_model.tag_tokenization import TagTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.sembert_featurizer import *
from pytorch_pretrained_bert.sembert_img_models import *
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm, trange
import torch
from sklearn.metrics import precision_recall_fscore_support
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from tag_model.modeling import TagConfig
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_DIR = "flickr30k_images"


class SnliImgProcessor(DataProcessor):
    """Processor for V-SNLI datasets"""
    
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv_tag_label")), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv_tag_label")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv_tag_label")), "test")
    
    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]
    
    def get_labels_map(self):
        return {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2
        }
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            
            if i == 0:
                continue
            guid = f"{set_type}-{i}-{line[0]}"
            if set_type == "test":
                text_a = line[-5]
                text_b = line[-4]
                img = line[-1]
                label = None
            else:
                text_a = line[-5]
                text_b = line[-4]
                img = line[-1]
                label = line[-3]
            
            example = InputImgExample(guid=guid, text_a=text_a, text_b=text_b, image=img, label=label)
            examples.append(example)
        return examples


class GSnliImgProcessor(DataProcessor):
    """Processor for V-SNLI datasets"""
    
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv_tag_label")), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv_tag_label")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv_tag_label")), "test")
    
    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]
    
    def get_labels_map(self):
        return {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2
        }
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        premises = []
        hypotheses = []
        for (i, line) in enumerate(lines):
            
            if i == 0:
                continue
            guid = f"{set_type}-{i}-{line[0]}"
            if set_type == "test":
                text_a = line[-4]
                text_b = line[-3]
                img = line[-2]
                label = None
            else:
                text_a = line[-4]
                text_b = line[-3]
                img = line[-2]
                label = line[-1]
            
            premise = InputImgExample(guid=guid, text_a=text_a, text_b=None, image=img, label=label)
            hypothesis = InputImgExample(guid=guid, text_a=text_b, text_b=None, image=img, label=label)
            premises.append(premise)
            hypotheses.append(hypothesis)
        return premises, hypotheses


class SequenceImageDataset(Dataset):
    """Custom dataset class"""
    
    def __init__(self, input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids, input_images,
                 transform, image_folder):
        self.transform = transform
        self.input_images = input_images
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_end_idx = start_end_idx
        self.input_tag_ids = input_tag_ids
        self.label_ids = label_ids
        self.image_folder = image_folder
    
    def __getitem__(self, index):
        try:
            image_file = self.input_images[index]
            # print(os.path.join(self.image_folder, image_file))
            PIL_image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image = self.transform(PIL_image)
        except:
            image = torch.zeros((3, 224, 224))
        input_id = torch.tensor(self.input_ids[index], dtype=torch.long)
        input_mask = torch.tensor(self.input_mask[index], dtype=torch.long)
        segment_id = torch.tensor(self.segment_ids[index], dtype=torch.long)
        start_end_idx = torch.tensor(self.start_end_idx[index], dtype=torch.long)
        input_tag_id = torch.tensor(self.input_tag_ids[index], dtype=torch.long)
        label_id = torch.tensor(self.label_ids[index], dtype=torch.long)
        
        return input_id, input_mask, segment_id, start_end_idx, input_tag_id, image, label_id
    
    def __len__(self):
        return len(self.label_ids)


class GroundedSequenceImageDataset(Dataset):
    """Custom dataset class"""
    
    def __init__(self, premise_input_ids, hypothesis_input_ids,
                 premise_input_mask, hypothesis_input_mask,
                 premise_segment_ids, hypothesis_segment_ids,
                 premise_start_end_idx, hypothesis_start_end_idx,
                 premise_input_tag_ids, hypothesis_input_tag_ids,
                 label_ids, input_images,
                 transform, image_folder):
        self.transform = transform
        self.input_images = input_images
        
        self.premise_input_ids = premise_input_ids
        self.hypothesis_input_ids = hypothesis_input_ids
        
        self.premise_input_mask = premise_input_mask
        self.hypothesis_input_mask = hypothesis_input_mask
        
        self.premise_segment_ids = premise_segment_ids
        self.hypothesis_segment_ids = hypothesis_segment_ids
        
        self.premise_start_end_idx = premise_start_end_idx
        self.hypothesis_start_end_idx = hypothesis_start_end_idx
        
        self.premise_input_tag_ids = premise_input_tag_ids
        self.hypothesis_input_tag_ids = hypothesis_input_tag_ids
        
        self.label_ids = label_ids
        self.image_folder = image_folder
    
    def __getitem__(self, index):
        try:
            image_file = self.input_images[index]
            # print(os.path.join(self.image_folder, image_file))
            PIL_image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image = self.transform(PIL_image)
        except:
            image = torch.zeros((3, 224, 224))
        
        premise_input_id = torch.tensor(self.premise_input_ids[index], dtype=torch.long)
        hypothesis_input_id = torch.tensor(self.hypothesis_input_ids[index], dtype=torch.long)
        input_id = torch.stack((premise_input_id, hypothesis_input_id))
        
        premise_input_mask = torch.tensor(self.premise_input_mask[index], dtype=torch.long)
        hypothesis_input_mask = torch.tensor(self.hypothesis_input_mask[index], dtype=torch.long)
        input_mask = torch.stack((premise_input_mask, hypothesis_input_mask))
        
        premise_segment_id = torch.tensor(self.premise_segment_ids[index], dtype=torch.long)
        hypothesis_segment_id = torch.tensor(self.hypothesis_segment_ids[index], dtype=torch.long)
        segment_id = torch.stack((premise_segment_id, hypothesis_segment_id))
        
        premise_start_end_idx = torch.tensor(self.premise_start_end_idx[index], dtype=torch.long)
        hypothesis_start_end_idx = torch.tensor(self.hypothesis_start_end_idx[index], dtype=torch.long)
        start_end_idx = torch.stack((premise_start_end_idx, hypothesis_start_end_idx))
        
        premise_input_tag_id = torch.tensor(self.premise_input_tag_ids[index], dtype=torch.long)
        hypothesis_input_tag_id = torch.tensor(self.hypothesis_input_tag_ids[index], dtype=torch.long)
        input_tag_id = torch.stack((premise_input_tag_id, hypothesis_input_tag_id))
        
        label_id = torch.tensor(self.label_ids[index], dtype=torch.long)
        
        return input_id, input_mask, segment_id, start_end_idx, input_tag_id, image, label_id
    
    def __len__(self):
        return len(self.label_ids)


class InputImgExample(InputExample):
    """A single training/test example for simple sequence and image classification"""
    
    def __init__(self, guid, text_a, image, text_b=None, label=None):
        super().__init__(guid, text_a, text_b, label)
        self.image = image


def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--data_dir",
                        default="data/VSNLI/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="snliimg",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="output_vsnli",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tagger_path", default=None, type=str,
                        help="tagger_path for predictions if needing real-time tagging. Default: None, by loading pre-tagged data"
                             "For example, the trained models by AllenNLP")
    parser.add_argument("--best_epochs",
                        default=1.0,
                        type=float,
                        help="Best training epochs for prediction.")
    parser.add_argument("--max_num_aspect",
                        default=3,
                        type=int,
                        help="max_num_aspect")
    
    ## Other parameters
    parser.add_argument("--grounding",
                        action='store_true',
                        help="whether to enable grounding.")
    parser.add_argument("--hypothesis_only",
                        action='store_true',
                        help="whether to enable grounding.")
    
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
        "snliimg": SnliImgProcessor,
        "gsnliimg": GSnliImgProcessor
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
    # num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.tagger_path != None:
        srl_predictor = SRLPredictor(args.tagger_path)
    else:
        srl_predictor = None
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        if args.grounding:
            train_premise_examples, train_hypothesis_examples = processor.get_train_examples(args.data_dir)
            train_examples = (train_premise_examples, train_hypothesis_examples)
            num_train_optimization_steps = int(
                len(
                    train_premise_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        else:
            train_examples = processor.get_train_examples(args.data_dir)
            num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    train_features = None
    if args.do_train:
        if args.grounding:
            hypothesis_features = convert_examples_to_features(
                train_examples[1], label_list, args.max_seq_length, tokenizer, srl_predictor=srl_predictor)
            
            premises_features = convert_examples_to_features(
                train_examples[0], label_list, args.max_seq_length, tokenizer, srl_predictor=srl_predictor)
            train_features = (premises_features, hypothesis_features)
        
        else:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, srl_predictor=srl_predictor)
        # TagTokenizer.make_tag_vocab("tag_vocab", tag_vocab)
    tag_tokenizer = TagTokenizer()
    vocab_size = len(tag_tokenizer.ids_to_tags)
    print("tokenizer vocab size: ", str(vocab_size))
    tag_config = TagConfig(tag_vocab_size=vocab_size,
                           hidden_size=10,
                           layer_num=1,
                           output_dim=10,
                           dropout_prob=0.1,
                           num_aspect=args.max_num_aspect)
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
                                                                   'distributed_{}'.format(args.local_rank))
    if args.grounding:
        if args.hypothesis_only:
            model = GroundedImgClassificationTag.from_pretrained(args.bert_model,
                                                                 cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                     args.local_rank),
                                                                 num_labels=num_labels, tag_config=tag_config,
                                                                 image_emb_size=2048, hypothesis_only=True)
        else:
            model = GroundedImgClassificationTag.from_pretrained(args.bert_model,
                                                                 cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                     args.local_rank),
                                                                 num_labels=num_labels, tag_config=tag_config,
                                                                 image_emb_size=2048)
    else:
        model = BertForSequenceImgClassificationTag.from_pretrained(args.bert_model,
                                                                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                        args.local_rank),
                                                                    num_labels=num_labels, tag_config=tag_config,
                                                                    image_emb_size=2048)
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    best_epoch = 0
    best_result = 0.0
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    if args.do_train:
        if not args.grounding:
            train_features = transform_tag_features(args.max_num_aspect, train_features, tag_tokenizer,
                                                    args.max_seq_length)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            # prepare data training data
            all_input_ids = [f.input_ids for f in train_features]
            all_input_mask = [f.input_mask for f in train_features]
            all_segment_ids = [f.segment_ids for f in train_features]
            all_label_ids = [f.label_id for f in train_features]
            all_start_end_idx = [f.orig_to_token_split_idx for f in train_features]
            all_input_tag_ids = [f.input_tag_ids for f in train_features]
            all_images = [f.image for f in train_features]
            train_data = SequenceImageDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx,
                                              all_input_tag_ids,
                                              all_label_ids, all_images, transform, IMAGE_DIR)
        
        else:
            premises_train_features = transform_tag_features(args.max_num_aspect, train_features[0], tag_tokenizer,
                                                             args.max_seq_length)
            hypothesis_train_features = transform_tag_features(args.max_num_aspect, train_features[1], tag_tokenizer,
                                                               args.max_seq_length)
            
            assert len(premises_train_features) == len(hypothesis_train_features)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples[0]))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            
            # prepare the premise training data
            all_premises_input_ids = [f.input_ids for f in premises_train_features]
            all_premises_input_mask = [f.input_mask for f in premises_train_features]
            all_premises_segment_ids = [f.segment_ids for f in premises_train_features]
            all_premises_start_end_idx = [f.orig_to_token_split_idx for f in premises_train_features]
            all_premises_input_tag_ids = [f.input_tag_ids for f in premises_train_features]

            # prepare the hypothesis training data
            all_hypothesis_input_ids = [f.input_ids for f in hypothesis_train_features]
            all_hypothesis_input_mask = [f.input_mask for f in hypothesis_train_features]
            all_hypothesis_segment_ids = [f.segment_ids for f in hypothesis_train_features]
            all_hypothesis_start_end_idx = [f.orig_to_token_split_idx for f in hypothesis_train_features]
            all_hypothesis_input_tag_ids = [f.input_tag_ids for f in hypothesis_train_features]
            
            all_images = [f.image for f in premises_train_features]
            all_label_ids = [f.label_id for f in premises_train_features]
            
            
            train_data = GroundedSequenceImageDataset(all_premises_input_ids, all_hypothesis_input_ids,
                                                      all_premises_input_mask, all_hypothesis_input_mask,
                                                      all_premises_segment_ids, all_hypothesis_segment_ids,
                                                      all_premises_start_end_idx, all_hypothesis_start_end_idx,
                                                      all_premises_input_tag_ids, all_hypothesis_input_tag_ids,
                                                      all_label_ids, all_images, transform, IMAGE_DIR)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        # prepare validation data
        
        if args.grounding:
            eval_premise_examples, eval_hypothesis_examples = processor.get_dev_examples(args.data_dir)
            eval_hypothesis_features = convert_examples_to_features(
                eval_hypothesis_examples, label_list, args.max_seq_length, tokenizer, srl_predictor=srl_predictor)
            
            eval_premises_features = convert_examples_to_features(
                eval_premise_examples, label_list, args.max_seq_length, tokenizer, srl_predictor=srl_predictor)
            
            eval_premises_features = transform_tag_features(args.max_num_aspect, eval_premises_features, tag_tokenizer,
                                                            args.max_seq_length)
            
            eval_hypothesis_features = transform_tag_features(args.max_num_aspect, eval_hypothesis_features,
                                                              tag_tokenizer,
                                                              args.max_seq_length)
            
            # prepare the premise training data
            all_premises_input_ids = [f.input_ids for f in eval_premises_features]
            all_premises_input_mask = [f.input_mask for f in eval_premises_features]
            all_premises_segment_ids = [f.segment_ids for f in eval_premises_features]
            all_premises_start_end_idx = [f.orig_to_token_split_idx for f in eval_premises_features]
            all_premises_input_tag_ids = [f.input_tag_ids for f in eval_premises_features]
            
            # prepare the hypothesis training data
            all_hypothesis_input_ids = [f.input_ids for f in eval_hypothesis_features]
            all_hypothesis_input_mask = [f.input_mask for f in eval_hypothesis_features]
            all_hypothesis_segment_ids = [f.segment_ids for f in eval_hypothesis_features]
            all_hypothesis_start_end_idx = [f.orig_to_token_split_idx for f in eval_hypothesis_features]
            all_hypothesis_input_tag_ids = [f.input_tag_ids for f in eval_hypothesis_features]
            
            all_images = [f.image for f in eval_hypothesis_features]
            all_label_ids = [f.label_id for f in eval_hypothesis_features]
            
            eval_data = GroundedSequenceImageDataset(all_premises_input_ids, all_hypothesis_input_ids,
                                                     all_premises_input_mask, all_hypothesis_input_mask,
                                                     all_premises_segment_ids, all_hypothesis_segment_ids,
                                                     all_premises_start_end_idx, all_hypothesis_start_end_idx,
                                                     all_premises_input_tag_ids, all_hypothesis_input_tag_ids,
                                                     all_label_ids, all_images, transform, IMAGE_DIR)
        
        
        
        
        else:
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
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, images, label_ids = batch
            
                loss = model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, images, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, str(epoch) + "_pytorch_model.bin")
            if args.do_train:
                torch.save(model_to_save.state_dict(), output_model_file)
            
            # run evaluation on dev data
            model_state_dict = torch.load(output_model_file)
            
            if not args.grounding:
                
                predict_model = BertForSequenceImgClassificationTag.from_pretrained(args.bert_model,
                                                                                    state_dict=model_state_dict,
                                                                                    num_labels=num_labels,
                                                                                    tag_config=tag_config,
                                                                                    image_emb_size=2048)
            else:
                predict_model = GroundedImgClassificationTag.from_pretrained(args.bert_model,
                                                                             state_dict=model_state_dict,
                                                                             num_labels=num_labels,
                                                                             tag_config=tag_config,
                                                                             image_emb_size=2048)
            
            predict_model.to(device)
            predict_model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            total_precision = np.zeros(3)
            total_recall = np.zeros(3)
            total_fscore = np.zeros(3)
            total_support = np.zeros(3, dtype=int)
            
            output_logits_file = os.path.join(args.output_dir, str(epoch) + "_eval_logits_results.tsv")
            with open(output_logits_file, "w") as writer:
                writer.write("index" + "\t" + "\t".join(["logits " + str(i) for i in range(len(label_list))]) + "\n")
                
                for batch_number, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                    input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, images, label_ids = batch
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
                    tmp_eval_accuracy = accuracy_score(label_ids, np.argmax(logits, axis=1), normalize=False)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy
                    
                    precision, recall, fscore, support = precision_recall_fscore_support(label_ids,
                                                                                         np.argmax(logits, axis=1),
                                                                                         labels=[0, 1, 2])
                    total_precision = total_precision + precision
                    total_recall = total_recall + recall
                    total_fscore = total_fscore + fscore
                    total_support = total_support + support
                    
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                
                del predict_model
                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                total_precision = total_precision / (batch_number + 1)
                total_recall = total_recall / (batch_number + 1)
                total_fscore = total_fscore / (batch_number + 1)
                
                if eval_accuracy > best_result:
                    best_epoch = epoch
                    best_result = eval_accuracy
                loss = tr_loss / nb_tr_steps if args.do_train else None
                
                result = {'eval_loss': eval_loss,
                          'loss': loss,
                          'eval_accuracy': eval_accuracy,
                          'total_precision': {
                              k: total_precision.tolist()[v] for k, v in processor.get_labels_map().items()
                          },
                          'total_recall': {
                              k: total_recall.tolist()[v] for k, v in processor.get_labels_map().items()
                          },
                          'total_fscore': {
                              k: total_fscore.tolist()[v] for k, v in processor.get_labels_map().items()
                          },
                          'total_support': {
                              k: total_support.tolist()[v] for k, v in processor.get_labels_map().items()
                          },
                          'macro_precision': total_precision.mean(),
                          'macro_recall': total_recall.mean(),
                          'macro_support': total_support.sum(),
                          'macro_f1score': total_fscore.mean(),
                          'number_of_examples': nb_eval_examples}
            
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
                    writer.write("Epoch: %s, %s = %s\n" % (str(epoch), key, str(result[key])))
        logger.info("best epoch: %s, result:  %s", str(best_epoch), str(best_result))


if __name__ == '__main__':
    main()