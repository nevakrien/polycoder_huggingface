import argparse
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, Trainer, TrainingArguments
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import os
import json

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokens=data['tokens']
        self.mask=data['mask']

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x=self.tokens[idx]
        mask=self.mask[idx]
        return {'input_ids': x, 'labels': x, 'attention_mask': mask}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #predictions = np.argmax(logits, axis=-1)
    #accuracy = (predictions == labels).mean()
    return {"lol": 0}

def main(args):
		config = GPTNeoXConfig.from_pretrained(args.config)
		model = GPTNeoXForCausalLM(config)

		train_data = np.load(os.path.join(args.data_dir, 'train_tokens.npz'))
		test_data = np.load(os.path.join(args.data_dir, 'test_tokens.npz'))
		train_dataset = TextDataset(train_data)
		test_dataset = TextDataset(test_data)

		optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
		scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.training_steps)


		training_args = TrainingArguments(
		    output_dir=args.save_dir,
		    per_device_train_batch_size=args.batch_size,
		    num_train_epochs=args.epochs,
		    save_strategy="epoch",
		    evaluation_strategy="steps",
		    eval_steps=args.eval_interval,
		    logging_dir='./logs'
		)

		trainer = Trainer(
		    model=model,
		    args=training_args,
		    train_dataset=train_dataset,
		    eval_dataset=test_dataset,
		    compute_metrics=compute_metrics,
		    optimizers=(optimizer, scheduler)
		)

		trainer.train()
		trainer.evaluate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Script to train GPT-2 model")

	parser.add_argument('--config', type=str, default='./configs/160m', help="Path to the config for building the model")
	parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the train and test data")
	parser.add_argument('--save_dir', type=str, required=True, help="Directory to save model checkpoints")
	parser.add_argument('--batch_size', type=int, default=262144, help="Big batch sizes are allowed (total #tokens per batch)")

	parser.add_argument('--lr', type=float, default=0.00016, help="Learning rate for the optimizer")
	parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train")
	parser.add_argument('--save_interval', type=int, default=1, help="Interval to save model checkpoints")
	parser.add_argument('--eval_interval', type=int, default=1, help="Interval to evaluate the model on test data")
	parser.add_argument('--warmup_steps', type=int, default=1600, help="Number of warmup steps")
	parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for the optimizer")
	parser.add_argument('--training_steps', type=int, default=150000, help="Total number of training steps")
	parser.add_argument('--adam_beta1', type=float, default=0.9, help="Beta1 for the Adam optimizer")
	parser.add_argument('--adam_beta2', type=float, default=0.999, help="Beta2 for the Adam optimizer")
	parser.add_argument('--adam_eps', type=float, default=1e-8, help="Epsilon for the Adam optimizer")

	args = parser.parse_args()
	main(args)