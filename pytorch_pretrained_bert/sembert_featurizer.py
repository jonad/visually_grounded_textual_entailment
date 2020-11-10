from .modeling import *


class SemBERTFeaturizer():
    def __init__(self, model_dir, bert_model, pretrained_model_file_name, tag_config):
        
        self.bert_model = bert_model
        self.pretrained_model_path = os.path.join(model_dir, pretrained_model_file_name)
        model_state_dict = torch.load(self.pretrained_model_path)
        self.tag_config = tag_config
        
        self.sembert_model = BertForSequenceClassificationTag.from_pretrained(self.bert_model,
                                                                              state_dict=model_state_dict,
                                                                              num_labels=3,
                                                                              tag_config=self.tag_config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sembert_model = self.sembert_model.to(self.device)
        self.sembert_model.eval()
    
    def __call__(self, data):
        
        for input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, _ in data:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            start_end_idx = start_end_idx.to(self.device)
            input_tag_ids = input_tag_ids.to(self.device)
            with torch.no_grad():
                sequence_output, _ = self.sembert_model.bert(input_ids, segment_ids, input_mask,
                                                             output_all_encoded_layers=False)
                batch_size, sub_seq_len, dim = sequence_output.size()
                start_end_idx = start_end_idx  # batch * seq_len * (start, end)
                max_seq_len = -1
                max_word_len = self.sembert_model.filter_size
                
                for se_idx in start_end_idx:
                    num_words = 0
                    for item in se_idx:
                        if item[0] != -1 and item[1] != -1:
                            num_subs = item[1] - item[0] + 1
                            if num_subs > max_word_len:
                                max_word_len = num_subs
                            num_words += 1
                    if num_words > max_seq_len:
                        max_seq_len = num_words
                assert max_word_len >= self.sembert_model.filter_size
            batch_start_end_ids = []
            batch_id = 0
            for batch in start_end_idx:
                word_seqs = []
                offset = batch_id * sub_seq_len
                for item in batch:
                    if item[0] != -1 and item[1] != -1:
                        subword_ids = list(range(offset + item[0] + 1, offset + item[1] + 2))
                        while len(subword_ids) < max_word_len:
                            subword_ids.append(0)
                        word_seqs.append(subword_ids)
                while (len(word_seqs) < max_seq_len):
                    word_seqs.append([0 for i in range(max_word_len)])
                batch_start_end_ids.append(word_seqs)
                batch_id += 1
            
            batch_start_end_ids = torch.tensor(batch_start_end_ids)
            batch_start_end_ids = batch_start_end_ids.view(-1)
            sequence_output = sequence_output.view(-1, dim)
            sequence_output = torch.cat([sequence_output.new_zeros((1, dim)), sequence_output], dim=0)
            batch_start_end_ids = batch_start_end_ids.cuda()
            cnn_bert = sequence_output.index_select(0, batch_start_end_ids)
            cnn_bert = cnn_bert.view(batch_size, max_seq_len, max_word_len, dim)
            cnn_bert = cnn_bert.cuda()
            bert_output = self.sembert_model.cnn(cnn_bert, max_word_len)
            num_aspect = input_tag_ids.size(1)
            input_tag_ids = input_tag_ids[:, :, :max_seq_len]
            flat_input_tag_ids = input_tag_ids.view(-1, input_tag_ids.size(-1))
            tag_output = self.sembert_model.tag_model(flat_input_tag_ids, num_aspect)
            tag_output = tag_output.transpose(1, 2).contiguous().view(batch_size,
                                                                      max_seq_len, -1)
            
            tag_output = self.sembert_model.dense(tag_output)
            sequence_output = torch.cat((bert_output, tag_output), 2)
            
            return sequence_output[:, 0]


if __name__ == '__main__':
    pass
