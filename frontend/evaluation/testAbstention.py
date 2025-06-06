# Run 'deepeval set-ollama mistral' in terminal for using Ollama

import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codes.EnchancedResponseChain import EnhancedResponseChain
from codes.ConfidenceCheck import ConfidenceChecker
from codes.GenericLLM import GenericLLM


class EnhancedAgent:
    def __init__(
        self,
        secrets,
        backend="ollama",
        llm_model_name="mistral",
        embed_model_name="nomic-embed-text",
        pine_index_name="td-bank-docs-new",
        use_cohere=True
    ):
        self.retriever = EnhancedResponseChain(
            secrets,
            backend,
            llm_model_name,
            embed_model_name,
            pine_index_name,
            use_cohere
        )

        # create an LLM based on the backend environment
        llm = GenericLLM(secrets, backend, llm_model_name)
        self.confidence_checker = ConfidenceChecker(llm, threshold=0.7)
        # next line is required because we implemented a custom LLM class (GenericLLM)
        # if the next line is not set, we get None errors
        self.confidence_checker.evaluation_cost = 0

    def invoke(self, query, conversation_history="", expected_output=None):
        """
        1. Get the answer from the retrieval pipeline.
        2. Check the answer using DeepEval-based confidence checker (which does NOT require expected_output).
        3. Return the final score.
        """
        result, top_context_vectors = self.retriever.answer_question_with_citations(
            query, conversation_history
        )

        final_answer, confidence_score, reason = self.confidence_checker.check_and_abstain(
            question=query,
            actual_output=result
        )
        return final_answer, confidence_score, reason


secrets_path = os.path.join(os.path.dirname(__file__), "../secrets_ollama.json")
with open(secrets_path, "r") as f:
    secrets = json.load(f)

#os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = secrets["pinecone_api_key"]

agent = EnhancedAgent(
    secrets=secrets,
    backend="ollama",
    llm_model_name="mistral",
    embed_model_name="nomic-embed-text",
    pine_index_name="td-bank-docs-new",
    use_cohere=True
)

# Read questions from the text file
questions_file_path = os.path.join(os.path.dirname(__file__), "test_questions.txt")
with open(questions_file_path, "r") as file:
    questions = file.readlines()

results = []

for question in questions:
    question = question.strip()
    if question:
        conversation_history = ""
        result, confidence_score, reason = agent.invoke(question, conversation_history)
        results.append({
            "Question": question,
            "Generated Answer": result,
            "Confidence Score": confidence_score,
            "Reason": reason,
            "abstained": confidence_score < 0.5
        })

output_file_path = os.path.join(os.path.dirname(__file__), "test_results.json")
with open(output_file_path, "w") as output_file:
    json.dump(results, output_file, indent=4)
