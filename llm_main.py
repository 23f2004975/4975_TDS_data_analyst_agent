from collections import defaultdict
import os, time
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_KEYS, MODEL_HIERARCHY
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]

class LLMWithFallback:
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)
        self.current_llm = None  # placeholder for actual ChatGoogleGenerativeAI instance

    def _get_llm_instance(self):
        last_error = None
        for model in self.models:
            for key in self.keys:
                try:
                    llm_instance = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key
                    )
                    self.current_llm = llm_instance
                    return llm_instance
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    if any(qk in msg for qk in QUOTA_KEYWORDS):
                        self.slow_keys_log[key].append(model)
                    self.failing_keys_log[key] += 1
                    time.sleep(0.5)
        raise RuntimeError(f"All models or the keys failed. Last error: {last_error}")

    def bind_tools(self, tools):
        llm_instance = self._get_llm_instance()
        return llm_instance.bind_tools(tools)


    def invoke(self, prompt):
        llm_instance = self._get_llm_instance()
        return llm_instance.invoke(prompt)

