import numpy as np
from transformers import ViTForImageClassification, TrainingArguments, Trainer, AutoImageProcessor
from datasets import load_dataset, DatasetDict, load_metric

class FontClassificationModel:
    def __init__(self, model_name_or_path, data_dir, train_size=0.7, validation_size=0.15, test_size=0.15):
        self.model_name_or_path = model_name_or_path
        self.data_dir = data_dir
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

    def load_dataset(self):
        ds = load_dataset("imagefolder", data_dir=self.data_dir)
        data_ds = ds['train'].train_test_split(shuffle=True, seed=0, test_size=self.train_size)
        data_test_ds = data_ds['test'].train_test_split(shuffle=True, seed=0, test_size=self.validation_size)
        final_dataset = DatasetDict({
            'train': data_ds['train'],
            'validation': data_test_ds['train'],
            'test': data_test_ds['test']
        })
        return final_dataset

    def transform(self, example_batch):
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name_or_path)
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        inputs['label'] = example_batch['label']
        return inputs

    def collate_fn(self, batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }

    def compute_metrics(self, p):
        metric = load_metric("accuracy")
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def train_model(self):
        prepared_ds = self.load_dataset().with_transform(self.transform)

        training_args = TrainingArguments(
            output_dir="./vit-base",
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=5,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
        )

        model = ViTForImageClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=len(prepared_ds['train'].features['label'].names),
            id2label={str(i): c for i, c in enumerate(prepared_ds['train'].features['label'].names)},
            label2id={c: str(i) for i, c in enumerate(prepared_ds['train'].features['label'].names)}
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=prepared_ds["train"],
            eval_dataset=prepared_ds["validation"],
        )

        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        output = trainer.predict(prepared_ds['test'])
        accuracy_score = self.compute_accuracy(output)
        return accuracy_score

    def compute_accuracy(self, output):
        def softmax(logits):
            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits)
            return probabilities

        target_labels = output.label_ids
        logits = output.predictions
        log_probs = softmax(logits)
        prediction_labels = np.argmax(log_probs, axis=-1)
        return accuracy_score(target_labels, prediction_labels)

# Usage
model_name_or_path = 'google/vit-base-patch16-224-in21k'
data_dir = "drive/MyDrive/data"
font_model = FontClassificationModel(model_name_or_path, data_dir)
accuracy = font_model.train_model()
print("Accuracy:", accuracy)
