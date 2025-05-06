import os
import pathlib
import re
import label_studio_sdk
import logging
import torch

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from transformers import pipeline, Pipeline
from itertools import groupby
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from datasets import Dataset, ClassLabel, Value, Sequence, Features
from functools import partial
import torch.nn as nn
import numpy as np
import random
import string
from uuid import uuid4

logger = logging.getLogger(__name__)
_model = None
_device = None
MODEL_DIR = os.getenv('MODEL_DIR', '/data/models')
BASELINE_MODEL_NAME = os.getenv('BASELINE_MODEL_NAME', 'Babelscape/wikineural-multilingual-ner')
FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', 'rmdr_ner_model.bin')
NUM_TAG = os.getenv('NUM_TAG', 10)
CLS = [101]
SEP = [102]
VALUE_TOKEN = [0]
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 8
EPOCHS = 15
SHIFT_SIZE = 50

class Dataset:
  
  def __init__(self, ds_raw):
    self.ds_raw = ds_raw

  def __len__(self):
    return len(self.ds_raw)

  def __getitem__(self, index):
  
    #ds_raw item = {'id': task['id'], 'tokens': [], 'tokens_ids': [], 'ner_tags': [], value: task['data'][value]}
    ds_raw_item = self.ds_raw[index]
    #Tokenise
    ids = ds_raw_item['tokens_ids']
    target_tag = ds_raw_item['ner_tags']
    
    #To Add Special Tokens, subtract 2 from MAX_LEN
    ids = ids[:MAX_LEN - 2]
    target_tag = target_tag[:MAX_LEN - 2]

    #Add Sepcial Tokens
    ids = CLS + ids + SEP
    target_tags = VALUE_TOKEN + target_tag + VALUE_TOKEN

    mask = [1] * len(ids)
    token_type_ids = [0] * len(ids)

    #Add Padding if the input_len is small

    padding_len = MAX_LEN - len(ids)
    ids = ids + ([0] * padding_len)
    target_tags = target_tags + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)

    return {
        "ids" : torch.tensor(ids, dtype=torch.long),
        "mask" : torch.tensor(mask, dtype=torch.long),
        "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
        "target_tags" : torch.tensor(target_tags, dtype=torch.long)
      }
  

class NERBertModel(nn.Module):
    
    def __init__(self, num_tag):
        super(NERBertModel, self).__init__()
        self.num_tag = num_tag
        self.bert = BertModel.from_pretrained(BASELINE_MODEL_NAME)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        
    def forward(self, ids, mask, token_type_ids, target_tags):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bert_out = self.bert_drop(output) 
        tag = self.out_tag(bert_out)
    
        #Calculate the loss
        Critirion_Loss = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = tag.view(-1, self.num_tag)
        active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags))
        loss = Critirion_Loss(active_logits, active_labels)
        return tag, loss



def reload_model():
    global _model, _tokenizer, _device
    _model = None
    _tokenizer = None
    _device = None
    try:
        logger.info(f"Получаем девайс...")
        _device =  "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Девайс... {_device}")
        
        # Создаем токинайзер для нашей модели
        logger.info(f"Loading _tokenizer from {BASELINE_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)
        logger.info(f"_tokenizer from {BASELINE_MODEL_NAME} loaded")
        
        # загружаем модель
        model_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
        logger.info(f"Loading finetuned model from {model_path}")
        _model = NERBertModel(num_tag=NUM_TAG)
        _model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(_device)))
        _model.eval()
        logger.info(f"...Loaded finetuned model from {model_path}")
    except Exception as e:
        # if finetuned model is not available, use the baseline model with the original labels
        logger.error(e)


reload_model()


class HuggingFaceNER(LabelStudioMLBase):
    """Custom ML Backend model
    """
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'c35a2f5689358d1e9d7522309643ba5b9cfca062')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))
    value = None

    def get_labels(self):
        li = self.label_interface
        from_name, _, _ = li.get_first_tag_occurence('Labels', 'Text')
        tag = li.get_tag(from_name)
        return tag.labels
    
    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
        from_name, to_name, self.value = self.label_interface.get_first_tag_occurence('Labels', 'Text')

    # Разбиваем тексты (списки токенов) которые больше 512 на куски меньше, со сдвигом назад на SHIFT_SIZE. 
    def make_ds_raw_ext (self, _ds_raw):
        ds_raw_ext = []
        for item in _ds_raw:
            tokens_len = len (item['tokens'])
            chank_index = 0
            for token_index in range(0, tokens_len, MAX_LEN-SHIFT_SIZE):
                token_index_end = token_index+MAX_LEN
                #print(token_index, token_index_end,  tokens_len, chank_index)
                item_ext = {'id': item['id'], 
                            'tokens': item['tokens'][token_index:token_index_end], 
                            'tokens_ids': item['tokens_ids'][token_index:token_index_end], 
                            'ner_tags': item['ner_tags'][token_index:token_index_end], 
                            self.value: item[self.value]}
                ds_raw_ext.append(item_ext)
                chank_index = chank_index + 1
        return ds_raw_ext

    # Предсказывает для заданного текста
    def prediction(self, sentence):
        
        ds_raw2 = []
        # now tokenize chunks separately and add them to the dataset
        Token_inputs = _tokenizer.tokenize(sentence)
        item = {'id': 0, 
                'tokens': Token_inputs, 
                'tokens_ids': _tokenizer.encode(sentence, add_special_tokens=False), 
                'ner_tags': [0] * len(Token_inputs), 
                self.value: sentence}
        ds_raw2.append(item)
        # Расширяем тексты (которые больше чем 512)
        ds_raw_ext2 = self.make_ds_raw_ext (ds_raw2)
        print(len(ds_raw2), len(ds_raw_ext2))
        test_dataset =  Dataset(ds_raw = ds_raw_ext2)
        tags = []
        scores = []
        with torch.no_grad():
            # Перебор датасета
            for index_data in range(test_dataset.__len__()):
                data = test_dataset[index_data]
                for i, j in data.items():
                    data[i] = j.to(_device).unsqueeze(0)
                tag, _ = _model(**data)
                tag2 = tag.argmax(2).cpu().numpy().reshape(-1)[1:len(Token_inputs)+1]
                score, max_indices  = torch.max(tag, dim=2)#.cpu().numpy().reshape(-1)[1:len(Token_inputs)+1]
                score = score.cpu().numpy().reshape(-1)[1:len(Token_inputs)+1]
                score_norm = (score - np.min(score))/np.ptp(score)
                tags.append (tag2)
                scores.append(score_norm)

        # Теперь нужно выровнять результат так как мы сделали это со сдвигом SHIFT_SIZE
        tags_result = []
        scores_result = []
        tokens_len = len (Token_inputs)
        print(tokens_len)
        chank_index = 0
        for token_index in range(0, tokens_len, MAX_LEN-SHIFT_SIZE):
            token_index_end = token_index+MAX_LEN
            tag = tags[chank_index]
            score = scores[chank_index]
            tag = np.pad(tag, (token_index, max(tokens_len - token_index - len(tag),0) ), 'constant', constant_values=(0, 0))
            tag = tag[0:tokens_len]
            tags[chank_index] = tag
            score = np.pad(score, (token_index, max(tokens_len - token_index - len(score),0) ), 'constant', constant_values=(0, 0))
            score = score[0:tokens_len]
            scores[chank_index] = score
            chank_index = chank_index + 1
        # теперь все массивы кол-во колонок - кол-во токенов
        tags_np = np.vstack( tags)
        # выбираем те теги и скоры, где скоры максимальные (на пересечении SHIFT_SIZE)
        scores_np = np.vstack( scores )
        scores_np_arg_max = np.argmax(scores_np, axis=0)
        tags_result = np.zeros(tokens_len)
        scores_result = np.zeros(tokens_len)
        for idxx, idxy in enumerate(scores_np_arg_max):
            tags_result[idxx] = tags_np[idxy, idxx]
            scores_result[idxx] = scores_np[idxy, idxx]
        
        tag2 = VALUE_TOKEN +list(tags_result) + VALUE_TOKEN
        score_norm = [0] + list(scores_result) + [0]
        return tag2, score_norm
    
    def generate_id(self, length=10):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for i in range(length))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # Получаем заданные метки слов
        labels = ['O'] + self.get_labels()
        id_to_label = {i: label for i, label in enumerate(labels)}
        label_to_id = {label: i  for i, label in enumerate(labels)}
        li = self.label_interface
        # Получаем данные (тексты)
        from_name, to_name, value = li.get_first_tag_occurence('Labels', 'Text')
        texts = [self.preload_task_data(task, task['data'][value]) for task in tasks]
        predictions = []
        for text in texts:
            logger.info(f"Текст: ")
            logger.info(text)
            # запускаем модель
            tags, score_norm = self.prediction(text)
            # токинизируем тестк (что бы сделать обратную разметку по словам)
            encoding = _tokenizer(text)
            tokens = encoding.tokens()
            word_index_prev = -1
            prev_label = 'O'
            predictions = []
            results = []
            avg_score = 0
            # перебираем полученные теги (и токены и скоре)
            for index in range(len(tags)):
                # получаем индекс слова по индексу токена
                word_index = encoding.token_to_word(index)
                # не для всех токенов есть слова
                if word_index != None:
                    #print (word_index)
                    # если слово новое (не повтор - одно слово может состоять из нескольких токенов)
                    if word_index_prev != word_index:
                        label = id_to_label[tags[index]]
                        score = score_norm[index]
                        if score >=0.1:
                            if label != 'O':
                                # начало и конец слова
                                start, end = encoding.word_to_chars(word_index) 
                                if prev_label != label:
                                    results.append({
                                            'id': str(uuid4()),#generate_id(),
                                            'from_name': from_name,
                                            'to_name': to_name,
                                            'type': 'labels',
                                            'value': {
                                                'start': start,
                                                'end': end,
                                                'labels': [label],
                                                'text': text[start:end],
                                                'token': tokens[index]
                                            },
                                            'score': score
                                        })
                                else:
                                    # Если метка уже была - то это многословная метка, значит просто продлеваем предыдущую
                                    #logger.info(f"results: {results}")
                                    if len(results) > 0:
                                        results[-1]["value"]["end"] = end
                                        results[-1]["value"]["text"] =  text[results[-1]["value"]["start"]:end]
                                avg_score += score
                        prev_label = label
                    word_index_prev = word_index
            if results:
                predictions.append({
                    'result': results,
                    'score': avg_score / len(results),
                    'model_version': self.get('model_version')
                })
        # Добавляем связи между тегами ['UNIT', 'WP', 'SLD', 'CAPT'] и тегом с количеством COUNT'
        # Исходим из того, что следующий после тега COUNT тег связан с этим COUNT
        #result_relations = []
        for index, item in enumerate ( predictions[0]['result']):
            #print(item)
            label = item['value']['labels'][0]
            if label == 'COUNT':
                # Берем следующий итем
                if index < len(predictions[0]['result']) - 1:
                    next_item = predictions[0]['result'][index+1]
                    next_label = next_item['value']['labels'][0]
                    if next_label in ['UNIT', 'WP', 'SLD', 'CAPT']:
                        # Добавляем связь между следющим итемом и итемом с числом
                        rel = { 'from_id': next_item['id'],
                                'to_id': item['id'],
                                'type': 'relation',
                                'direction': 'right',
                                #'labels': ["has_count"],
                                'region': item
                            }
                        relations = []
                        relations.append(rel)
                        next_item['relations'] = relations
                        #result_relations.append(rel)
        # if result_relations and predictions:
        #     predictions[0]['result'].append(result_relations)
        # Возвращаем результат для Label Studio        
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

    def _get_tasks(self, project_id):
        # download annotated tasks from Label Studio
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        return tasks

    def tokenize_and_align_labels(self, examples, tokenizer):
        """
        From example https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess
        """
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def fit(self, event, data, **kwargs):
        # """Download dataset from Label Studio and prepare data for training in BERT
        # """
        # if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
        #     logger.info(f"Skip training: event {event} is not supported")
        #     return

        # project_id = data['annotation']['project']
        # tasks = self._get_tasks(project_id)

        # if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
        #     logger.info(f"Skip training: {len(tasks)} tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}")
        #     return

        # # we need to convert Label Studio NER annotations to hugingface NER format in datasets
        # # for example:
        # # {'id': '0',
        # #  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
        # #  'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
        # # }
        # ds_raw = []
        # from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
        # tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)

        # no_label = 'O'
        # label_to_id = {no_label: 0}
        # for task in tasks:
        #     for annotation in task['annotations']:
        #         if not annotation.get('result'):
        #             continue
        #         spans = [{'label': r['value']['labels'][0], 'start': r['value']['start'], 'end': r['value']['end']} for r in annotation['result']]
        #         spans = sorted(spans, key=lambda x: x['start'])
        #         text = self.preload_task_data(task, task['data'][value])

        #         # insert tokenizer.pad_token to the unlabeled chunks of the text in-between the labeled spans, as well as to the beginning and end of the text
        #         last_end = 0
        #         all_spans = []
        #         for span in spans:
        #             if last_end < span['start']:
        #                 all_spans.append({'label': no_label, 'start': last_end, 'end': span['start']})
        #             all_spans.append(span)
        #             last_end = span['end']
        #         if last_end < len(text):
        #             all_spans.append({'label': no_label, 'start': last_end, 'end': len(text)})

        #         # now tokenize chunks separately and add them to the dataset
        #         item = {'id': task['id'], 'tokens': [], 'ner_tags': []}
        #         for span in all_spans:
        #             tokens = tokenizer.tokenize(text[span['start']:span['end']])
        #             item['tokens'].extend(tokens)
        #             if span['label'] == no_label:
        #                 item['ner_tags'].extend([label_to_id[no_label]] * len(tokens))
        #             else:
        #                 label = 'B-' + span['label']
        #                 if label not in label_to_id:
        #                     label_to_id[label] = len(label_to_id)
        #                 item['ner_tags'].append(label_to_id[label])
        #                 if len(tokens) > 1:
        #                     label = 'I-' + span['label']
        #                     if label not in label_to_id:
        #                         label_to_id[label] = len(label_to_id)
        #                     item['ner_tags'].extend([label_to_id[label] for _ in range(1, len(tokens))])
        #         ds_raw.append(item)

        # logger.debug(f"Dataset: {ds_raw}")
        # # convert to huggingface dataset
        # # Define the features of your dataset
        # features = Features({
        #     'id': Value('string'),
        #     'tokens': Sequence(Value('string')),
        #     'ner_tags': Sequence(ClassLabel(names=list(label_to_id.keys())))
        # })
        # hf_dataset = Dataset.from_list(ds_raw, features=features)
        # tokenized_dataset = hf_dataset.map(partial(self.tokenize_and_align_labels, tokenizer=tokenizer), batched=True)

        # logger.debug(f"HF Dataset: {tokenized_dataset}")

        # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        # id_to_label = {i: label for label, i in label_to_id.items()}
        # logger.debug(f"Labels: {id_to_label}")

        # model = AutoModelForTokenClassification.from_pretrained(
        #     BASELINE_MODEL_NAME, num_labels=len(id_to_label),
        #     id2label=id_to_label, label2id=label_to_id)
        # logger.debug(f"Model: {model}")

        # training_args = TrainingArguments(
        #     output_dir=str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME),
        #     learning_rate=self.LEARNING_RATE,
        #     per_device_train_batch_size=8,
        #     num_train_epochs=self.NUM_TRAIN_EPOCHS,
        #     weight_decay=self.WEIGHT_DECAY,
        #     evaluation_strategy="no",
        # )

        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=tokenized_dataset,
        #     tokenizer=tokenizer,
        #     data_collator=data_collator,
        # )
        # trainer.train()

        # chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
        # logger.info(f"Model is trained and saved as {chk_path}")
        # trainer.save_model(chk_path)

        # # reload model
        # # TODO: this is not thread-safe, should be done with critical section
        reload_model()
