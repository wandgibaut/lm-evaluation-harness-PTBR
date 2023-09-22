import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


class AGNews(Task):
    VERSION = 0
    DATASET_PATH = "maritaca-ai/ag_news_pt"
    # DATASET_PATH = "/content/drive/MyDrive/Artigos/Cabrita/ag_news/agnews_DEBUG"
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
            doc["text"].strip()
            + ("" if doc["text"].strip().endswith(".") else "."),
        )
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " {}".format({0:"mundo", 1:"esportes", 2:"negocios", 3:"tecnologia"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_mundo,        _ = rf.loglikelihood(ctx, " mundo")
        ll_esportes,     _ = rf.loglikelihood(ctx, " esportes")
        ll_negócios,     _ = rf.loglikelihood(ctx, " negocios")
        ll_tecnologia,   _ = rf.loglikelihood(ctx, " tecnologia")
        return ll_mundo, ll_esportes, ll_negócios, ll_tecnologia

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}