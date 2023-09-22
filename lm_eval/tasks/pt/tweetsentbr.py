import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_score
from functools import partial


class TweetSentBR(Task):
    VERSION = 0
    DATASET_PATH = "/content/drive/MyDrive/Artigos/Cabrita/tweetsentBR/tweetSentBR"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return "Texto: {}\nClasse:".format(
            doc["tweet"].strip()
            + ("" if doc["tweet"].strip().endswith(".") else "."),
        )
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["tweet"]

    def doc_to_target(self, doc):
        return " {}".format({0:"negativa", 1:"positiva", 2:"neutra"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_positiva, _ = rf.loglikelihood(ctx, " positiva")
        ll_negativa, _ = rf.loglikelihood(ctx, " negativa")
        ll_neutra,   _ = rf.loglikelihood(ctx, " neutra")
        return ll_negativa, ll_positiva, ll_neutra

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold,
                "f1_macro": (gold, pred)}

    def aggregation(self):
        return {"acc": mean, 
                "f1_macro": partial(
                    f1_score, average="macro")
                }

    def higher_is_better(self):
        return {"acc": True, "f1_macro": True}