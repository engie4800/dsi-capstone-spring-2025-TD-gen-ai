from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

from .GEvalStrict import GEvalStrict


class ConfidenceChecker:
    def __init__(self, llm: DeepEvalBaseLLM, threshold: float = 0.5):
        self.threshold = threshold
        #ge = GEval(
        ge = GEvalStrict(
            name="CorrectnessNoExpectedOutput",
            criteria="Assess whether the actual output is consistent with or does not contradict the input.",
            # evaluation_steps=[
            #
            #     "Check whether 'actual output' clearly contradicts the 'input'",
            #     "If the input implies certain factual info, see if the 'actual output' is consistent with that",
            #     "Minor omissions or ambiguities might lower the score slightly",
            #     "Return a score between 0 and 1, where 1 indicates the question was answered correctly and 0 indicates it was answered incorrectly or was abstained."
            #     "If the model returns an answer that contains 'I'm not confident enough in my answer.' always return 0."
            # ],
            # evaluation_steps=[
            #     "Check whether the actual output directly answers the question.",
            #     "If the question asks for specific data and the output fails to provide it, lower the score.",
            #     "If the actual output evades the question or provides general information instead of an answer, assign a low score.",
            #     "Only assign a high score (close to 1) if the actual output fully and directly answers the question.",
            #     "Verbose explanations that do not provide the requested answer should lower the score.",
            #     "Assign a score of 0 if the output is irrelevant, evasive, or hallucinated."
            # ],
            evaluation_steps=[
                "Does the actual output answer the specific question? If not, assign a low score.",
                "If the actual output provides surrounding or general context but fails to directly answer the question, assign a score of 0.",
                "If the question cannot be answered due to missing information, and the actual output clearly states this *without adding distracting speculation*, assign a score up to 0.5.",
                "If the actual output fabricates information or misleads with approximations or tangents, assign a score of 0.",
                "Only assign a score near 1.0 if the actual output gives a direct and correct answer to the question."
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=llm
        )
        ge.evaluation_cost = 0
        self.correctness_metric = ge

    def measure_confidence(self, question: str, actual_output: str) -> tuple[float, str]:
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            #expected_output=actual_output  # Optional: can be None if your metric doesn’t use it
        )
        try:
            result = self.correctness_metric.evaluate(test_case)
            #print(f"Result={result}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        return result

    def check_and_abstain(self, question: str, actual_output: str):
        score, reason = self.measure_confidence(question, actual_output)
        #print(f"score & reason: {score} {reason}")
        return actual_output, score, reason
