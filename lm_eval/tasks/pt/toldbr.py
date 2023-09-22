import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_score
from functools import partial


class ToLDBR(Task):
    VERSION = 0
    DATASET_PATH = "told-br"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["binary"]["train"])
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["binary"]["test"]
        
    def validation_docs(self):
        return self.dataset["binary"]["validation"]

    def doc_to_text(self, doc):
        return "Texto: {}\nClasse:".format(
            doc["text"].strip()
            + ("" if doc["text"].strip().endswith(".") else "."),
        )
    #return f"{doc['passage']}\nQuestão: {doc['question']}?\nResposta:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " {}".format({0:"not-toxic", 1:"toxic"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_toxic, _ = rf.loglikelihood(ctx, " toxic")
        ll_not_toxic, _ = rf.loglikelihood(ctx, " not-toxic")
        return ll_not_toxic, ll_toxic

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