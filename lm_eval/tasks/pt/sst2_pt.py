import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean

class SST2(Task):
    VERSION = 1
    DATASET_PATH = "maritaca-ai/sst2_pt"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "Texto: {}\nClasse:".format(
            doc["text"].strip()
            + ("" if doc["text"].strip().endswith(".") else "."),
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " {}".format({0:"negativa", 1:"positiva"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_negativa, _ = rf.loglikelihood(ctx, " negativa")
        ll_positiva, _ = rf.loglikelihood(ctx, " positiva")
        return ll_negativa, ll_positiva


    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


