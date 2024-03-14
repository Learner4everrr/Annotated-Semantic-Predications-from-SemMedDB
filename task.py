import argparse
import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import DataCollatorWithPadding
import evaluate

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM

from model_creator import model_creator


parser = argparse.ArgumentParser(description='config')
parser.add_argument("-model", default='bert-base-uncased', choices=["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "google-bert/bert-large-uncased"],
	type=str, help="Choose the model and corresponding tokenizer")  
parser.add_argument("-epoch", type=int, help="choose epoch number", required=True)
parser.add_argument("-random", default=42, type=int, help="choose epoch number", required=True)
args = parser.parse_args()


def pre_processing(example):
    sentence = example['SENTENCE']
    subject = example['SUBJECT_TEXT']
    _object = example['OBJECT_TEXT']
    relation = example['PREDICATE']
    # text = f"{subject} [SEP] {relation} [SEP] {object} [SEP] {sentence}"
    text = f"{sentence} [SEP] {subject} , {relation} , {_object}"
    return text

def save_result(result):
	file_name = 'result.txt'
	with open(file_name, 'a+') as file:
		file.write(result)


def main():
	print("model name:", args.model)
	print("epoch number:", args.epoch)
	print("random seed for spliting dataset:", args.random)


	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f'device: {device}')

	df = pd.read_csv('substance_interactions.csv')
	df['triple_with_sentence'] = df.apply(pre_processing,axis=1)


	model, tokenizer = model_creator(args.model)

	# print("1"*100)

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	accuracy = evaluate.load('accuracy')

	# def compute_metrics(eval_pred):
	#     predictions, labels = eval_pred
	#     predictions = np.argmax(predictions, axis=1)
	#     return accuracy.compute(predictions=predictions, references=labels)
	    
	def compute_metrics(eval_pred):
	    predictions, labels = eval_pred
	    predictions = np.argmax(predictions, axis=1)
	    
	    # Calculate precision, recall, and F1 score
	    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
	    
	    return {
	        'accuracy': accuracy_score(labels, predictions),
	        'precision': precision,
	        'recall': recall,
	        'f1': f1
	    }


	def processing(example):
	    res = tokenizer(example['triple_with_sentence'])
	    # res['label'] = example['LABEL']
	    res['label'] = 1 if example['LABEL']=='y' else 0
	    return res


	df['data'] = df.apply(processing, axis=1)

	train_data, test_data = train_test_split(df, test_size=0.3, random_state=args.random)
	val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=args.random)
	train_data = train_data.reset_index()
	val_data = val_data.reset_index()
	test_data = test_data.reset_index()



	training_args = TrainingArguments(
	    output_dir='my_best_model',
	    learning_rate=2e-5,
	    per_device_train_batch_size=32,
	    per_device_eval_batch_size=32,
	    num_train_epochs=args.epoch,
	    weight_decay=0.01,
	    evaluation_strategy='epoch',
	    save_strategy='epoch',
	    load_best_model_at_end=True
	)

	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=train_data['data'],
	    eval_dataset=val_data['data'],
	    tokenizer=tokenizer,
	    data_collator=data_collator,
	    compute_metrics=compute_metrics
	)

	trainer.train()

	res1 = trainer.evaluate(train_data['data'])

	res2 = trainer.evaluate(val_data['data'])

	res3 = trainer.evaluate(test_data['data'])
	# print(res2)

	save_result("\n\nResults on Training set:\n model name:{}\nepoch number:{}\nrandom seed for spliting dataset:{}\n".format(args.model,args.epoch,args.random) + str(res1))

	save_result("\n\nResults on Val set:\n model name:{}\nepoch number:{}\nrandom seed for spliting dataset:{}\n".format(args.model,args.epoch,args.random) + str(res2))

	save_result("\n\nResults on test set:\n model name:{}\nepoch number:{}\nrandom seed for spliting dataset:{}\n".format(args.model,args.epoch,args.random) + str(res3))




if __name__ == '__main__':
    main()