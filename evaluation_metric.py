import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from typing import List

class BLEUEvaluationMetric:
    """
    This class is used to evaluate the quality of generated output sequences using the BLEU score.
    """

    def __init__(self):
        """
        Initializes the BLEUEvaluationMetric class.
        """
        pass

    def compute_bleu(self, predictions: List[List[str]], references: List[List[List[str]]]) -> float:
        """
        Computes the BLEU score for the given predictions and references.

        Args:
        predictions (List[List[str]]): A list of predicted output sequences.
        references (List[List[List[str]]]): A list of reference output sequences.

        Returns:
        float: The computed BLEU score.
        """
        return corpus_bleu(references, predictions)

    def evaluate(self, predictions: List[List[str]], references: List[List[List[str]]]) -> float:
        """
        Evaluates the quality of the generated output sequences using the BLEU score.

        Args:
        predictions (List[List[str]]): A list of predicted output sequences.
        references (List[List[List[str]]]): A list of reference output sequences.

        Returns:
        float: The computed BLEU score.
        """
        return self.compute_bleu(predictions, references)

if __name__ == "__main__":
    evaluation_metric = BLEUEvaluationMetric()
    predictions = [["This", "is", "a", "test"], ["This", "is", "another", "test"]]
    references = [[["This", "is", "a", "test"]], [["This", "is", "another", "test"]]]
    bleu_score = evaluation_metric.evaluate(predictions, references)
    print(f"BLEU Score: {bleu_score}")