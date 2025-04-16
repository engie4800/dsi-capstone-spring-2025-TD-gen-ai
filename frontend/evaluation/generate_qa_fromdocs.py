import sys
import json
import re
from pathlib import Path


def read_chunks(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def generate_answerable_prompt(text_chunk, num_questions):
    return f"""
You are a helpful assistant that generates factual questions.

Given the following text chunk:
\"\"\"
{text_chunk}
\"\"\"

Generate {num_questions} questions that are clearly **answerable** based on this text. 
Avoid yes/no questions. Make the questions specific and focused on content that is actually in the text.

Respond only with a JSON list of strings (questions). No explanations.
"""


def generate_unanswerable_prompt(text_chunk, num_questions):
    return f"""
You are a helpful assistant that generates challenging comprehension questions.

Given the following text chunk:
\"\"\"
{text_chunk}
\"\"\"

Generate {num_questions} questions that are **unanswerable** based on this text. 
Do not include yes/no questions. Each question should seem plausible but not be answerable from the provided text.

Respond only with a JSON list of strings (questions). No explanations.
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
        print("Usage: python generate_questions.py <backend> <llm_model_name> <text_file> <num_answerable> <num_unanswerable> [chunk_size]")
        sys.exit(1)

    backend = sys.argv[1]
    llm_model_name = sys.argv[2]
    text_file = sys.argv[3]
    num_answerable = int(sys.argv[4])
    num_unanswerable = int(sys.argv[5])
    chunk_size = int(sys.argv[6]) if len(sys.argv) >= 7 else 20000

    print(f"Generating questions using backend {backend} with llm {llm_model_name}")

    # Load secrets info from appropriate backend secrets file
    secrets_filename = f"../secrets_{backend}.json"
    with open(secrets_filename) as f:
        secrets = json.load(f)

    llm = load_llm(backend, llm_model_name, secrets)

    all_answerable = []
    all_unanswerable = []

    for i, chunk in enumerate(read_chunks(text_file, chunk_size), start=1):
        print(f"🔹 Processing chunk {i}...")

        ans_prompt = generate_answerable_prompt(chunk, num_answerable)
        unans_prompt = generate_unanswerable_prompt(chunk, num_unanswerable)

        try:
            answerable = ask_llm(ans_prompt, llm)
            unanswerable = ask_llm(unans_prompt, llm)
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")
            continue

        all_answerable.extend(answerable)
        all_unanswerable.extend(unanswerable)

    with open("answerable_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_answerable, f, indent=2, ensure_ascii=False)

    with open("unanswerable_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_unanswerable, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_answerable)} answerable questions to answerable_questions.json")
    print(f"✅ Saved {len(all_unanswerable)} unanswerable questions to unanswerable_questions.json")

if __name__ == "__main__":
    import os
    main()
