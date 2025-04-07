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

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.llm.invoke(prompt)
        #print(f"\nresponse={response}")
        #return response.content
        return response

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def generate_raw_response(self, prompt: str, **kwargs):
        response_text = self.generate(prompt, **kwargs)
        self._token_cost += 0  # dummy cost, but ensures int is used

        return response_text, 0

        # Load JSON string into a dictionary
        data = json.loads(response_text)

        # Add a new property
        data['cost'] = 0

        # Convert dictionary back to JSON string
        new_json_str = json.dumps(data)

        #data['content'] = data[]


        print(f"\n>>>{data}")
        #return new_json_str,0
        return data, 0

    def get_model_name(self) -> str:
        return self.llm_model_name

    @property
    def token_cost(self):
        return self._token_cost
