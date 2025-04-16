import json

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI, AzureChatOpenAI

class GenericLLM(DeepEvalBaseLLM):
    def __init__(self, secrets, backend='ollama', llm_model_name="mistral"):
        self.secrets = secrets
        self.llm_model_name = llm_model_name
        self.backend = backend
        self.llm = None
        self._token_cost = 0  # private variable to track cost
        self.evaluation_cost = 0
        self.load_model()

    def load_model(self):
        if self.backend == "openai":
            self.llm = ChatOpenAI(
                model=self.llm_model_name,
                temperature=0,
                api_key=self.secrets["openai_api_key"],
                openai_api_base=self.secrets["openai_api_endpoint"],
            )
        elif self.backend == "azure":
            self.llm = AzureChatOpenAI(
                model=self.llm_model_name,
                temperature=0,
                max_retries=2,
                azure_endpoint=self.secrets["openai_api_endpoint"],
                api_key=self.secrets["openai_api_key"],
                openai_api_version=self.secrets["openai_api_version"]
            )
        elif self.backend == "ollama":
            self.llm = ChatOpenAI(
                model=self.llm_model_name,
                temperature=0,
                api_key=self.secrets["openai_api_key"],
                openai_api_base=self.secrets["openai_api_endpoint"],
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # def generate(self, prompt: str, **kwargs) -> str:
    #     response = self.llm.invoke(prompt)
    #     #print(f"\nresponse={response}")
    #     #return response.content
    #     return response

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.llm.invoke(prompt)
        return response.content  # ✅ Required: extract text

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    # def generate_raw_response(self, prompt: str, **kwargs):
    #     response_text = self.generate(prompt, **kwargs)
    #     self._token_cost += 0  # dummy cost, but ensures int is used
    #
    #     return response_text, 0
    #
    #     # Load JSON string into a dictionary
    #     data = json.loads(response_text)
    #
    #     # Add a new property
    #     data['cost'] = 0
    #
    #     # Convert dictionary back to JSON string
    #     new_json_str = json.dumps(data)
    #
    #     #data['content'] = data[]
    #
    #
    #     print(f"\n>>>{data}")
    #     #return new_json_str,0
    #     return data, 0

    # def generate_raw_response(self, prompt: str, **kwargs):
    #     response_text = self.generate(prompt, **kwargs)
    #     self._token_cost += 0
    #     return response_text, 0  # ✅ Ensure this is a (str, float)

    # def generate_raw_response(self, prompt: str, **kwargs):
    #     response_text = self.generate(prompt, **kwargs)
    #
    #     try:
    #         parsed = json.loads(response_text)
    #         score = parsed.get("score")
    #         if isinstance(score, (int, float)) and 0 <= score <= 1:
    #             pass
    #         else:
    #             print(f"⚠️ Invalid score format in LLM response: {score}")
    #     except Exception:
    #         print("⚠️ LLM did not return valid JSON: ", response_text)
    #
    #     return response_text, 0

    def generate_raw_response(self, prompt: str, **kwargs):
        raw_response = self.generate(prompt, **kwargs)
        self._token_cost += 0

        # Strip markdown code blocks if needed
        response_text = raw_response.strip()
        if response_text.startswith("```json"):
            response_text = response_text.removeprefix("```json").removesuffix("```").strip()
        elif response_text.startswith("```"):
            response_text = response_text.removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(response_text)

            score = parsed.get("score", 0)
            if not isinstance(score, (int, float)):
                print(f"⚠️ Score is not a number: {score}, defaulting to 0.0")
                parsed["score"] = 0.0
            elif not (0.0 <= float(score) <= 1.0):
                print(f"⚠️ Invalid score {score} — clamping to range [0.0, 1.0]")
                parsed["score"] = max(0.0, min(1.0, float(score)))

            class MockResponse:
                def __init__(self, content):
                    self.content = json.dumps(content)

            return MockResponse(parsed), 0

        except Exception as e:
            print(f"❌ Failed to parse JSON from LLM response:\n{raw_response}\nError: {e}")
            fallback = {
                "score": 0.0,
                "reason": "LLM response was not valid JSON."
            }

            class MockResponse:
                def __init__(self, content):
                    self.content = json.dumps(content)

            return MockResponse(fallback), 0

    def get_model_name(self) -> str:
        return self.llm_model_name

    @property
    def token_cost(self):
        return self._token_cost
