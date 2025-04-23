import sys
import json
import re
import random
from pathlib import Path

def read_chunks(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_answerable_prompt(text_chunk):
    return f"""
You are a helpful assistant that generates factual questions.

Given the following text chunk:
\"\"\"
{text_chunk}
\"\"\"

Generate **one** question and answer that is clearly answerable based on this text. 
Avoid yes/no questions. Make the question specific and focused on content that is actually in the text.

Respond only with a JSON array with one object, using this exact format:
[
  {{
    "Question": "<question>",
    "Answer": "<answer>"
  }}
]
"""

def ask_llm(prompt, llm):
    response = llm.generate(prompt)
    try:
        return json.loads(response)
    except Exception:
        match = re.search(r"\[(.*?)\]", response, re.DOTALL)
        if match:
            try:
                return json.loads("[" + match.group(1) + "]")
            except Exception:
                pass
        raise ValueError("LLM did not return a valid JSON list of questions.")

def load_llm(backend, model_name, secrets=None):
    from deepeval.models import DeepEvalBaseLLM
    from langchain_openai import ChatOpenAI, AzureChatOpenAI

    class GenericLLM(DeepEvalBaseLLM):
        def __init__(self, backend, model_name, secrets=None):
            self.backend = backend
            self.llm_model_name = model_name
            self.secrets = secrets or {}
            self.evaluation_cost = 0
            self._token_cost = 0
            self.load_model()

        def load_model(self):
            if self.backend == "openai":
                self.llm = ChatOpenAI(
                    model=self.llm_model_name,
                    temperature=0,
                    api_key=self.secrets.get("openai_api_key"),
                    openai_api_base=self.secrets.get("openai_api_endpoint"),
                )
            elif self.backend == "azure":
                self.llm = AzureChatOpenAI(
                    model=self.llm_model_name,
                    temperature=0,
                    api_key=self.secrets.get("openai_api_key"),
                    azure_endpoint=self.secrets.get("openai_api_endpoint"),
                    openai_api_version=self.secrets.get("openai_api_version"),
                )
            elif self.backend == "ollama":
                self.llm = ChatOpenAI(
                    model=self.llm_model_name,
                    temperature=0,
                    api_key=self.secrets.get("openai_api_key"),
                    openai_api_base=self.secrets.get("openai_api_endpoint"),
                )
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        def generate(self, prompt: str, **kwargs) -> str:
            response = self.llm.invoke(prompt)
            return response.content

        async def a_generate(self, prompt: str, **kwargs) -> str:
            return self.generate(prompt, **kwargs)

        def generate_raw_response(self, prompt: str, **kwargs):
            result = self.generate(prompt, **kwargs)
            self._token_cost += 0
            return result, 0

        def get_model_name(self) -> str:
            return self.llm_model_name

        @property
        def token_cost(self):
            return self._token_cost

    return GenericLLM(backend, model_name, secrets)

def main():
    if len(sys.argv) < 6:
        print("Usage: python generate_questions.py <backend> <llm_model_name> <text_file> <num_answerable_questions> [chunk_size]")
        sys.exit(1)

    backend = sys.argv[1]
    llm_model_name = sys.argv[2]
    text_file = sys.argv[3]
    num_answerable_questions = int(sys.argv[4])
    chunk_size = int(sys.argv[5]) if len(sys.argv) >= 6 else 20000

    print(f"Generating {num_answerable_questions} questions using backend {backend} with llm {llm_model_name}")

    secrets_filename = f"../secrets_{backend}.json"
    with open(secrets_filename) as f:
        secrets = json.load(f)

    llm = load_llm(backend, llm_model_name, secrets)
    chunks = read_chunks(text_file, chunk_size)

    all_answerable = []
    attempts = 0

    while len(all_answerable) < num_answerable_questions:
        chunk = random.choice(chunks)
        attempts += 1
        print(f"🔹 Attempt {attempts}: Sampling and generating from random chunk...{len(all_answerable)}")

        ans_prompt = generate_answerable_prompt(chunk)
        try:
            answerable = ask_llm(ans_prompt, llm)
            #print(answerable)
            if answerable and isinstance(answerable, list):
                all_answerable.extend(answerable[:1])
        except Exception as e:
            print(f"❌ Failed on attempt {attempts}: {e}")

    with open("answerable_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_answerable[:num_answerable_questions], f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_answerable[:num_answerable_questions])} answerable questions to answerable_questions.json")

if __name__ == "__main__":
    main()
