'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from pcgrad import PCGrad
from gradvac_amp import GradVacAMP
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

from tokenizer import BertTokenizer

TQDM_DISABLE=False

torch.cuda.empty_cache()
# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
                
        # Sentiment classification layer
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # Paraphrase detection layer
        # self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE*2, 1)
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE, 1)

        # Semantic Textual Similarity layer
        # self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE*2, 1)
        self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE, 1)
        
        # Dropout layer
        self.dropout_sentiment = nn.Dropout(config.hidden_dropout_prob)

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
        # encode the input_ids and attention_mask using the BERT model
        outputs = self.bert(input_ids, attention_mask)
        classification_embedding = outputs['pooler_output']

        return classification_embedding

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        
        # encode the input_ids and attention_mask using the BERT model
        # outputs = self.bert(input_ids, attention_mask)
        classification_embeddings = self.forward(input_ids, attention_mask)
        classification_embeddings = self.dropout_sentiment(classification_embeddings)
        sentiment_logits = self.sentiment_classifier(classification_embeddings)

        return sentiment_logits
        
    
    def combine_sentences(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''
        Combines a pair of sentences into a single embedding.
        '''
        
        # Get the token ID for the [SEP] token
        SEP_TOKEN_ID = self.tokenizer.sep_token_id

        # Create a tensor for the [SEP] token ID
        sep_token_id = torch.tensor([SEP_TOKEN_ID], dtype=torch.long, device=input_ids_1.device)

        # Repeat the [SEP] token ID to match the batch size of the input sentences
        repeated_sep_tokens = sep_token_id.repeat(input_ids_1.shape[0], 1)

        # Combine the input sentences with separator tokens in between
        # sentance1 [SEP] sentance2 [SEP]
        # Concatenate token IDs of the first sentence, separator tokens, token IDs of the second sentence, and separator tokens
        input_id = torch.cat((input_ids_1, repeated_sep_tokens, input_ids_2, repeated_sep_tokens), dim=1)

        # Create a tensor of ones the same size as the repeated_sep_tokens tensor for the SEP attention mask
        sep_attention_mask = torch.ones_like(repeated_sep_tokens)

        # create a global attention mask
        global_attention_mask = torch.cat( 
            (attention_mask_1, sep_attention_mask, attention_mask_2, sep_attention_mask)
            , dim=1)
        
        return input_id, global_attention_mask
    
    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        
        input_id, global_attention_mask = self.combine_sentences(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

        # encode the input_ids and attention_mask using the BERT model
        class_embed = self.forward(input_id, global_attention_mask)

        # pass the combined classification embedding through the paraphrase classifier
        paraphrase_logits = self.paraphrase_classifier(class_embed)
        # TODO: add more hidden layers here

        return paraphrase_logits

        # encode the input_ids and attention_mask using the BERT model
        # class_embed_1 = self.forward(input_ids_1, attention_mask_1)
        # class_embed_2 = self.forward(input_ids_2, attention_mask_2)

        # # dropout the classification embeddings
        # class_embed_1 = self.dropout_sentiment(class_embed_1)
        # class_embed_2 = self.dropout_sentiment(class_embed_2)

        # combined_cls_embedding = torch.cat((class_embed_1, class_embed_2), dim=1)

        # # pass the combined classification embedding through the paraphrase classifier
        # paraphrase_logits = self.paraphrase_classifier(combined_cls_embedding)

        # return paraphrase_logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''

        input_id, global_attention_mask = self.combine_sentences(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

        # encode the input_ids and attention_mask using the BERT model
        class_embed = self.forward(input_id, global_attention_mask)

        # pass the combined classification embedding through the paraphrase classifier
        paraphrase_logits = self.similarity_classifier(class_embed)
        # TODO: add more hidden layers here

        return paraphrase_logits

        # class_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        # class_embedding_2 = self.forward(input_ids_2, attention_mask_2)

        # # dropout the classification embeddings
        # class_embedding_1 = self.dropout_sentiment(class_embedding_1)
        # class_embedding_2 = self.dropout_sentiment(class_embedding_2)
        # combined_class_embedding = torch.cat((class_embedding_1, class_embedding_2), dim=1)

        # similarity_logits = self.similarity_classifier(combined_class_embedding)

        # return similarity_logits
    

def save_model(model, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def get_train_batch(name, iterations, iterator_dataloaders): 
    '''Get next batch from iterators'''
    try: 
        return next(iterations[name])
    except: 
        iterations[name] = iter(iterator_dataloaders[name])
        return next(iterations[name])

def process_batch(task, iterators, iterator_dataloaders, batch_size, device, model, weight):
    '''Process a batch accoring to the task'''
    batch = get_train_batch(task, iterators, iterator_dataloaders)
    
    if task == "sst": 
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)
        outputs = model.bert(b_ids, b_mask)
        embeddings = outputs['pooler_output']
        def predict_sentiment_pertrubed(embed):
            classification_embeddings = model.dropout_sentiment(embed)
            sentiment_logits = model.sentiment_classifier(classification_embeddings)
            return sentiment_logits
        smart_loss_fn = SMARTLoss(eval_fn = predict_sentiment_pertrubed, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        logits = predict_sentiment_pertrubed(embeddings)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size
        loss +=  weight * smart_loss_fn(embeddings, logits) / batch_size
        return loss

    elif task == "para":
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                              batch['attention_mask_1'],
                                                              batch['token_ids_2'],
                                                              batch['attention_mask_2'],
                                                              batch['labels'])
        b_ids_1 = b_ids_1.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_labels = b_labels.to(device)
        input_id, global_attention_mask = model.combine_sentences(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        outputs = model.bert(input_id, global_attention_mask)
        embeddings = outputs['pooler_output']
        def predict_para_pertrub(embed):
            paraphrase_logits = model.paraphrase_classifier(embed)
            return paraphrase_logits
        smart_loss_fn = SMARTLoss(eval_fn = predict_para_pertrub, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        logits = predict_para_pertrub(embeddings)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float(), reduction='sum') / args.batch_size
        loss +=  weight * smart_loss_fn(embeddings, logits) / batch_size
        return loss
    else: 
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                              batch['attention_mask_1'],
                                                              batch['token_ids_2'],
                                                              batch['attention_mask_2'],
                                                              batch['labels'])
        b_ids_1 = b_ids_1.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_labels = b_labels.to(device)
        input_id, global_attention_mask = model.combine_sentences(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        outputs = model.bert(input_id, global_attention_mask)
        embeddings = outputs['pooler_output']
        def predict_sts_pertrub(embed):
            similarity_logits = model.similarity_classifier(embed)
            return similarity_logits
        smart_loss_fn = SMARTLoss(eval_fn = predict_sts_pertrub, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        loss = F.mse_loss(logits.view(-1).float(), b_labels.view(-1).float(), reduction='sum') / args.batch_size
        loss +=  weight * smart_loss_fn(embeddings, logits) / batch_size
    return loss



def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print(device)
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=int(args.batch_size),
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=int(args.batch_size),
                                    collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,   
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    sst_iteration = iter(sst_train_dataloader)
    para_iteration = iter(para_train_dataloader)
    sts_iteration = iter(sts_train_dataloader)
    iterators = {"sst": sst_iteration, "para": para_iteration, "sts": sts_iteration}
    iterator_dataloaders = {"sst": sst_train_dataloader, "para": para_train_dataloader, "sts": sts_train_dataloader}
    accumulation_steps = args.accum_steps//args.batch_size
    pc_optimizer = PCGrad(optimizer)
    scaler = torch.cuda.amp.GradScaler()
    print(args.use_vac)
    weight = args.smart_weight
    print(weight)
    grad_vac_optimizer = GradVacAMP(3, optimizer, device, scaler = scaler, beta = 1e-2, reduction='sum', cpu_offload = False)
    for epoch in range(3):
        model.train()
        iterator_batch_nums = {"sst": 0, "para": 0, "sts": 0}
        iterator_batch_losses = {"sst": 0, "para": 0, "sts": 0}
        for i in tqdm(range(len(sts_train_dataloader)), desc=f'Train {epoch}', disable=TQDM_DISABLE, smoothing=0):
            losses = []
            for task in ["para"]:
                loss_task = process_batch(task, iterators, iterator_dataloaders, args.batch_size, device, model, weight)
                iterator_batch_nums[task] += 1
                losses.append(loss_task)
                iterator_batch_losses[task] += loss_task.item()
            if args.use_vac:
                grad_vac_optimizer.backward(losses)
            else: 
                pc_optimizer.pc_backward(losses)
            if args.use_vac:
                if (i + 1) % accumulation_steps == 0:
                    grad_vac_optimizer.step()
                    grad_vac_optimizer.zero_grad()
            else:
                pc_optimizer.step()
                pc_optimizer.zero_grad()
            torch.cuda.empty_cache()

        dev_sentiment_accuracy,_, _, \
            dev_paraphrase_accuracy, _, _, \
            dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)
        
        for task in ["para"]:
            print(f"Epoch {epoch} {task}: train loss :: {iterator_batch_losses[task]/iterator_batch_nums[task] :.3f}")
        print(f"Epoch {epoch} (sst): dev acc :: {dev_sentiment_accuracy :.3f}") 
        print(f"Epoch {epoch} (para): dev acc :: {dev_paraphrase_accuracy :.3f}")
        print(f"Epoch {epoch} (sts): dev acc :: {dev_sts_corr :.3f}")
        mean_dev = (dev_sentiment_accuracy + dev_paraphrase_accuracy + dev_sts_corr)/3
        
        if mean_dev > best_dev_acc:
            best_dev_acc = mean_dev
            save_model(model, args, config, args.filepath)
    for epoch in range(args.epochs):
        model.train()
        iterator_batch_nums = {"sst": 0, "para": 0, "sts": 0}
        iterator_batch_losses = {"sst": 0, "para": 0, "sts": 0}
        for i in tqdm(range(len(sts_train_dataloader)), desc=f'Train {epoch}', disable=TQDM_DISABLE, smoothing=0):
            losses = []
            for task in ["sst", "para", "sts"]:
                loss_task = process_batch(task, iterators, iterator_dataloaders, args.batch_size, device, model, weight)
                iterator_batch_nums[task] += 1
                losses.append(loss_task)
                iterator_batch_losses[task] += loss_task.item()
            if args.use_vac:
                grad_vac_optimizer.backward(losses)
            else: 
                pc_optimizer.pc_backward(losses)
            if args.use_vac:
                if (i + 1) % accumulation_steps == 0:
                    grad_vac_optimizer.step()
                    grad_vac_optimizer.zero_grad()
            else:
                pc_optimizer.step()
                pc_optimizer.zero_grad()
            torch.cuda.empty_cache()

        dev_sentiment_accuracy,_, _, \
            dev_paraphrase_accuracy, _, _, \
            dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)
        
        for task in ["sst", "para", "sts"]:
            print(f"Epoch {epoch} {task}: train loss :: {iterator_batch_losses[task]/iterator_batch_nums[task] :.3f}")
        print(f"Epoch {epoch} (sst): dev acc :: {dev_sentiment_accuracy :.3f}") 
        print(f"Epoch {epoch} (para): dev acc :: {dev_paraphrase_accuracy :.3f}")
        print(f"Epoch {epoch} (sts): dev acc :: {dev_sts_corr :.3f}")
        mean_dev = (dev_sentiment_accuracy + dev_paraphrase_accuracy + dev_sts_corr)/3
        
        if mean_dev > best_dev_acc:
            best_dev_acc = mean_dev
            save_model(model, args, config, args.filepath)
  
    
def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--accum_steps", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--use_vac", action='store_true')
    parser.add_argument("--smart_weight", type=float, help="weight for smart loss", default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
