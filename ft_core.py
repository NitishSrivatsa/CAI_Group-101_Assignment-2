from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

DEFAULT_BASE = "google/flan-t5-small"

class FTClient:
    def __init__(self, base_model: str = DEFAULT_BASE,
                 adapter_repo: Optional[str] = None, adapter_filename: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.adapter_loaded = False
        if adapter_repo and adapter_filename:
            try:
                local_path = hf_hub_download(repo_id=adapter_repo, filename=adapter_filename)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
                self.adapter_loaded = True
            except Exception:
                self.adapter_loaded = False

    def answer(self, question: str, context: str, max_new_tokens: int = 128) -> str:
        prompt = (
            "You are a financial reporting assistant. "
            "Answer concisely and only using the given context. "
            "If the answer is not in the context, say 'Data not in scope.'\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
