from transformers import Seq2SeqTrainer
import math
import logging


logger = logging.getLogger(__name__)
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSeq2SeqTrainer, self).__init__(*args, **kwargs)
        print("Load CustomSeq2SeqTrainer...")
        self.step_count = 0

    def _set_requires_grad(self, epoch):
        """
        Enable or disable layers for training based on the epoch number.
        """
                
        if epoch == 0:
            print("******init model***********")
            for name, param in self.model.named_parameters():
                if 'lora_B' in name:
                    # print(name)
                    param.requires_grad = True
                elif 'lora_A' in name:
                    param.requires_grad = False
        elif epoch != 0 and math.floor(epoch%20) == 0:
            print("Freeze the matrix")
            for name, param in self.model.named_parameters():
                if 'lora_A' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                    elif not param.requires_grad:
                        param.requires_grad = True
                elif 'lora_B' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                    elif not param.requires_grad:
                        # print(name)
                        param.requires_grad = True

    def training_step(self, model, inputs):
        """
        Perform a training step with the given model and inputs.
        """
        # Modify requires_grad attribute at the beginning of each epoch
        # epoch_number = int(self.state.global_step / len(self.get_train_dataloader()))
        # self.step_count += 1
        self._set_requires_grad(self.step_count)
        self.step_count += 1

        # Proceed with the regular training step
        return super(CustomSeq2SeqTrainer, self).training_step(model, inputs)

# # Instantiate the custom trainer
# model = ...  # Your Seq2Seq model
# args = Seq2SeqTrainingArguments(...)
# data_collator = ...  # Your data collator if needed
# train_dataset = ...  # Your training dataset

# trainer = CustomSeq2SeqTrainer(
#     model=model,
#     args=args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
#     # Other arguments as needed
# )

# # Start training
# trainer.train()
