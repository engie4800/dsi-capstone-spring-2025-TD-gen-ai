from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

class ConfidenceChecker:
    def __init__(self, llm: DeepEvalBaseLLM, threshold: float = 0.5):
        self.threshold = threshold
        ge = GEval(
            name="CorrectnessNoExpectedOutput",
            criteria="Assess whether the actual output is consistent with or does not contradict the input.",
            evaluation_steps=[
                "Check whether 'actual output' clearly contradicts the 'input'",
                "If the input implies certain factual info, see if the 'actual output' is consistent with that",
                "Minor omissions or ambiguities might lower the score slightly"
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        return result

    def check_and_abstain(self, question: str, actual_output: str):
        score, reason = self.measure_confidence(question, actual_output)
        #print(f"score & reason: {score} {reason}")
        return actual_output, score, reason
