from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from utils.dataPreprocess import DataPreprocess
import torch

class DataModeling(DataPreprocess):
    def __init__(self, oldfolder, newfolder):
        super().__init__(oldfolder, newfolder)
        self.custom_config = GPT2Config(
            activation_function='relu',
            n_positions=1024,  
            n_embd=768,        
            n_layer=12,        
            n_head=12,         
            resid_pdrop=0.1,   
            embd_pdrop=0.1,   
            attn_pdrop=0.1,
            vocab_size=self.tokenizer.vocab_size
        )

        self.model = GPT2LMHeadModel(self.custom_config)

    def calc_loss_batch(self, input_batch, target_batch, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device) #A
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
        return loss
    
    def calc_loss_loader(self, data_loader, device, num_batches=None):
        total_loss = 0.

        if len(data_loader) == 0:
            return float("nan")
        
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch, device)
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches
    
    def TrainingModel(
        self, 
        output_dir='/kaggle/working/small', 
        evaluation_strategy='epoch',
        epoch=10, 
        num_batch=3, 
        train_dataset=[], 
        val_dataset=[],
        lr=0.01,
        weight_decay=0.01
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model.to(device)  

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            learning_rate=lr,
            save_steps='epoch',
            per_device_train_batch_size=num_batch,
            per_device_eval_batch_size=num_batch,
            num_train_epochs=epoch,
            weight_decay=weight_decay,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Using the same dataset for evaluation
        )

        trainer.train() 
        
        return trainer
    
    def testModel(self, dir, input_text):
        # Load the fine-tuned GPT-2 model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(dir)
        tokenizer = GPT2Tokenizer.from_pretrained(dir)
        tokenizer.pad_token = tokenizer.eos_token

        # Send the model to the GPU (if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        input_ids = tokenizer(input_text, return_tensors='pt').to(device)

        output = model.generate(
            input_ids['input_ids'],
            max_length=100,  
            num_return_sequences=1,  # Number of sequences to generate
            no_repeat_ngram_size=2,  # To avoid repeating n-grams
            early_stopping=True,
            attention_mask=input_ids['attention_mask'],
            pad_token_id=tokenizer.pad_token_id 
        )

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return f'Generate text: {output_text}'