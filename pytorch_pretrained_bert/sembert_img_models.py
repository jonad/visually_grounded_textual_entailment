from .modeling import *


class BertForSequenceImgTag(BertForSequenceClassificationTag):
    def __init__(self, config, num_labels=2, tag_config=None, image_emb_size=None):
        super(BertForSequenceImgTag, self).__init__(config, num_labels=num_labels, tag_config=tag_config)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        self.image_emb_size = image_emb_size
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.cnn_img = nn.Sequential(nn.Linear(image_emb_size, 1024), nn.Linear(1024, 512))
        
        self.pool = nn.Linear(config.hidden_size + tag_config.hidden_size + 512,
                              config.hidden_size + tag_config.hidden_size)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, input_tag_ids=None,
                image=None, labels=None):
 
        
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        no_cuda = False
        batch_size, sub_seq_len, dim = sequence_output.size()
        # sequence_output = sequence_output.unsqueeze(1)
        start_end_idx = start_end_idx  # batch * seq_len * (start, end)
        max_seq_len = -1
        max_word_len = self.filter_size
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
        assert max_word_len >= self.filter_size
        batch_start_end_ids = []
        batch_id = 0
        for batch in start_end_idx:
            word_seqs = []
            offset = batch_id * sub_seq_len
            for item in batch:
                if item[0] != -1 and item[1] != -1:
                    subword_ids = list(range(offset + item[0] + 1, offset + item[1] + 2))  # 0用来做padding了
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
        if not no_cuda:
            batch_start_end_ids = batch_start_end_ids.cuda()
        cnn_bert = sequence_output.index_select(0, batch_start_end_ids)
        cnn_bert = cnn_bert.view(batch_size, max_seq_len, max_word_len, dim)
        if not no_cuda:
            cnn_bert = cnn_bert.cuda()
        
        bert_output = self.cnn(cnn_bert, max_word_len)
        
        num_aspect = input_tag_ids.size(1)
        input_tag_ids = input_tag_ids[:, :, :max_seq_len]
      
        flat_size = input_tag_ids.size(-1)
       
        # flat_input_tag_ids = input_tag_ids.view(-1, input_tag_ids.size(-1))
        flat_input_tag_ids = input_tag_ids.reshape(-1, flat_size)
        
        # print("flat_que_tag", flat_input_que_tag_ids.size())
        tag_output = self.tag_model(flat_input_tag_ids, num_aspect)
        # batch_size, que_len, num_aspect*tag_hidden_size
        tag_output = tag_output.transpose(1, 2).contiguous().view(batch_size,
                                                                  max_seq_len, -1)
        tag_output = self.dense(tag_output)
        sequence_output = torch.cat((bert_output, tag_output), 2)
        # print("tag", tag_output.size())
        # print("bert", bert_output.size())
        
        # print(image)
        img_sequence_output = self.resnet(image)
        img_sequence_output = img_sequence_output.view(img_sequence_output.size(0), -1)
        img_output = self.cnn_img(img_sequence_output)
        
        # print(img_sequence_output.size())
        img_embeddings = img_output.view(img_output.size(0), -1)
        bert_tag_embeddings = sequence_output[:, 0]
        return bert_tag_embeddings, img_embeddings


class BertForSequenceImgClassificationTag(BertForSequenceImgTag):
    def __init__(self, config, num_labels=2, tag_config=None, image_emb_size=None):
        super(BertForSequenceImgClassificationTag, self).__init__(config, num_labels, tag_config, image_emb_size)
        
        self.pool = nn.Linear(config.hidden_size + tag_config.hidden_size + 512,
                              config.hidden_size + tag_config.hidden_size)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, input_tag_ids=None,
                image=None, labels=None):
        
        first_token_tensor, img_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask,
                                                         start_end_idx=start_end_idx, input_tag_ids=input_tag_ids,
                                                         image=image)
        first_token_tensor = first_token_tensor.cuda()
        img_output = img_output.cuda()
        
        img_sequence_tensor = torch.cat((first_token_tensor, img_output), 1)
        
        # print(img_sequence_tensor.size())
        pooled_output = self.pool(img_sequence_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class GroundedImgClassificationTag(BertForSequenceImgTag):
    
    def __init__(self, config, num_labels=2, tag_config=None, image_emb_size=None, hypothesis_only=False):
        super(GroundedImgClassificationTag, self).__init__(config, num_labels, tag_config, image_emb_size)
        self.pool = nn.Linear(config.hidden_size * 2 + tag_config.hidden_size * 2 + 512 * 2,
                              config.hidden_size + tag_config.hidden_size)
        
        self.hypothesis_only = hypothesis_only
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_end_idx=None, input_tag_ids=None, image=None, labels=None):
        
        
        
        premise_bert_tag_embeddings, img_embeddings = super().forward(input_ids[:, 0],
                                                                      token_type_ids[:, 0],
                                                                      attention_mask[:, 0],
                                                                      start_end_idx[:, 0],
                                                                      input_tag_ids[:, 0], image)
        
        hypothesis_bert_tag_embeddings, _ = super().forward(input_ids[:, 1],
                                                            token_type_ids[:, 1],
                                                            attention_mask[:, 1],
                                                            start_end_idx[:, 1],
                                                            input_tag_ids[:, 1], image)
        
        img_embeddings = img_embeddings.cuda()
        premise_bert_tag_embeddings = premise_bert_tag_embeddings.cuda()
        hypothesis_bert_tag_embeddings = hypothesis_bert_tag_embeddings.cuda()
        if self.hypothesis_only:
            grounded_premise_tensor = torch.cat(
                (premise_bert_tag_embeddings, torch.zeros(img_embeddings.size()).cuda()), 1)
        
        else:
            grounded_premise_tensor = torch.cat((premise_bert_tag_embeddings, img_embeddings), 1)
        
        grounded_hypothesis_tensor = torch.cat((hypothesis_bert_tag_embeddings, img_embeddings), 1)
        img_sequence_tensor = torch.cat((grounded_premise_tensor, grounded_hypothesis_tensor), 1)
        img_sequence_tensor = self.pool(img_sequence_tensor)
        
        pooled_output = self.activation(img_sequence_tensor)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
           
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

# class GroundedImgClassificationTag(BertForSequenceClassificationTag):
#     def __init__(self, config, num_labels=2, tag_config=None, image_emb_size=None):
#         super(BertForSequenceImgClassificationTag, self).__init__(config, num_labels, tag_config, image_emb_size)
#
#         self.bert_for_fequence_img_tag = BertForSequenceImgTag(config=config, tag_config=tag_config,
#                                                                image_emb_size=image_emb_size)
#
#
#         self.pool = nn.Linear(config.hidden_size*2 + tag_config.hidden_size*2 + 512*2,
#                                    config.hidden_size + tag_config.hidden_size)
#
#     def forward(self, premise_input_ids, hypothesis_input_ids, premise_token_type_ids=None, premise_attention_mask=None,
#                 premise_start_end_idx=None, premise_input_tag_ids=None,
#                  hypothesis_token_type_ids=None, hypothesis_attention_mask=None,
#                 hypothesis_start_end_idx=None, hypothesis_input_tag_ids=None,labels=None, image=None, hypothesis_only=False):
#
#         premise_bert_tag_embeddings, img_embeddings = self.bert_for_seq_img(premise_input_ids, premise_token_type_ids, premise_attention_mask,
#                                                                     premise_start_end_idx, premise_input_tag_ids, image)
#
#         hypothesis_bert_tag_embeddings, _  = self.bert_for_seq_img(hypothesis_input_ids,hypothesis_token_type_ids,
#                                                                                     hypothesis_attention_mask,
#                                                                                     hypothesis_start_end_idx,
#                                                                                     hypothesis_input_tag_ids, image)
#
#
#         if hypothesis_only:
#             grounded_premise_tensor = torch.cat((premise_bert_tag_embeddings, torch.zeros(img_embeddings.size())), 1)
#
#         else:
#             grounded_premise_tensor = torch.cat((premise_bert_tag_embeddings, img_embeddings), 1)
#
#         grounded_hypothesis_tensor = torch.cat((hypothesis_bert_tag_embeddings, img_embeddings), 1)
#         img_sequence_tensor = torch.cat((grounded_premise_tensor, grounded_hypothesis_tensor), 1)
#         img_sequence_tensor = self.pool_grounded(img_sequence_tensor)
#
#         pooled_output = self.activation(img_sequence_tensor)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits

