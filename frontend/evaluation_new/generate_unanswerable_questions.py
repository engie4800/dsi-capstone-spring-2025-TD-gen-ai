import sys
import json
import re



def generate_unanswerable_prompt(num_questions):
    return f"""
You are a helpful assistant that generates challenging financial questions.
Generate exactly {num_questions} distinct unanswerable questions that are similar in style 
to questions about TD Bank's publicly available financial reports, but refer to details 
or ask for information that the reports do not provide. Avoid yes/no questions. Always use 
valid dates and years if the question references a date. Use the
answer 'This information is not available.' for all unanswerable questions.

Respond only with a JSON array with one object, using this exact format:
[
  {{
    "Question": "<question>",
    "Answer": "This information is not available"
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
    if len(sys.argv) < 4:
        print("Usage: python generate_questions.py <backend> <llm_model_name> <num_unanswerable>")
        sys.exit(1)

    backend = sys.argv[1]
    llm_model_name = sys.argv[2]
    num_unanswerable = int(sys.argv[3])

    print(f"Generating questions using backend {backend} with llm {llm_model_name}")

    # Load secrets info from appropriate backend secrets file
    secrets_filename = f"../secrets_{backend}.json"
    with open(secrets_filename) as f:
        secrets = json.load(f)

    llm = load_llm(backend, llm_model_name, secrets)

    all_unanswerable = []

    unans_prompt = generate_unanswerable_prompt(num_unanswerable)

    try:
        all_unanswerable = ask_llm(unans_prompt, llm)
    except Exception as e:
        print(f"❌ Failed on chunk {i}: {e}")


    with open("unanswerable_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_unanswerable, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_unanswerable)} unanswerable questions to unanswerable_questions.json")

if __name__ == "__main__":
    import os
    main()
