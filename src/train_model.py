import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HateTweetModel:
    """
    A model class for fine-tuning and evaluating a RoBERTa-based model for hate tweet detection.
    
    Attributes:
        model (RobertaForSequenceClassification): The pre-trained RoBERTa model.
    """
    
    def __init__(self, model_name='roberta-base', num_labels=2):
        """
        Initialize the HateTweetModel with a pre-trained RoBERTa model.

        Args:
            model_name (str): The name of the pre-trained model to load.
            num_labels (int): The number of output labels for classification.
        """
        logging.info(f"Initializing model: {model_name} with {num_labels} labels.")
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def compute_metrics(self, eval_pred):
        """
        Compute accuracy, precision, recall, and F1-score for evaluation.

        Args:
            eval_pred (tuple): A tuple containing model predictions and actual labels.

        Returns:
            dict: A dictionary containing accuracy, F1-score, precision, and recall.
        """
        logits, labels = eval_pred
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        predictions = torch.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.numpy(), average='binary')
        acc = accuracy_score(labels, predictions.numpy())

        logging.info(f"Metrics - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self, train_dataset, eval_dataset, output_dir='./results', epochs=3, batch_size=16):
        """
        Fine-tune the RoBERTa model on the given training dataset.

        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
            output_dir (str): Directory to save the model and logs.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and evaluation.

        Returns:
            RobertaForSequenceClassification: The trained RoBERTa model.
        """
        logging.info(f"Training started for {epochs} epochs with batch size {batch_size}.")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        self.save_model(output_dir)
        logging.info("Training completed.")
        return self.model

    def save_model(self, output_dir):
        """
        Save the trained model to the specified directory.

        Args:
            output_dir (str): Directory to save the model.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")

    def evaluate_and_save_metrics(self, test_dataset, output_dir='./results'):
        """
        Evaluate the model on the test dataset and save the metrics and confusion matrix.

        Args:
            test_dataset: The dataset for testing the model.
            output_dir (str): Directory to save the metrics and confusion matrix images.
        """
        logging.info("Evaluating the model on the test dataset.")
        trainer = Trainer(model=self.model)
        predictions, labels, _ = trainer.predict(test_dataset)

        # Calculate metrics
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

        logging.info(f"Test metrics - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        
        # Save metrics and confusion matrix
        metrics = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
        self.plot_and_save_metrics(metrics, output_dir)
        self.plot_and_save_confusion_matrix(labels, predictions, output_dir)

    def plot_and_save_metrics(self, metrics, output_dir):
        """
        Plot and save evaluation metrics as bar charts.

        Args:
            metrics (dict): Dictionary of evaluation metrics.
            output_dir (str): Directory to save the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red'])
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Score')
        plt.savefig(os.path.join(output_dir, 'metrics.png'))
        plt.close()
        logging.info(f"Metrics plot saved to {output_dir}/metrics.png")

    def plot_and_save_confusion_matrix(self, labels, predictions, output_dir):
        """
        Plot and save the confusion matrix.

        Args:
            labels (array-like): True labels.
            predictions (array-like): Model predictions.
            output_dir (str): Directory to save the plot.
        """
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        logging.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


# Data processor and usage

from data_preprocessing import DataProcessor

logging.info("Starting data processing...")

processor = DataProcessor(train_file='/home/abu/Documents/Hate Tweet Analysis/data/train.csv',
                          test_file='/home/abu/Documents/Hate Tweet Analysis/data/test.csv')

train_data, val_data = processor.prepare_train_val_data()
test_data = processor.prepare_test_data()

# Initialize and train the model
hate_model = HateTweetModel()
model = hate_model.train(train_data, val_data)

# Evaluate on test data and save metrics
hate_model.evaluate_and_save_metrics(test_data, output_dir='/home/abu/Documents/Hate Tweet Analysis/results')

logging.info("Model evaluation and metrics saving completed.")
