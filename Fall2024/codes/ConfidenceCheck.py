from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

class ConfidenceChecker:
    """
    A simple confidence checker module that compute a confidence score for a given response. This score can be used to decide whether to 'abstain' or return the response.
    """

    """
    A confidence checker that uses DeepEval's GEval to compare actual_output and expected_output.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

        # Define a GEval metric that only looks at INPUT and ACTUAL_OUTPUT
        self.correctness_metric = GEval(
            name="CorrectnessNoExpectedOutput",
            criteria="Assess whether the actual output is consistent with or does not contradict the input.",
            evaluation_steps=[
                "Check whether 'actual output' clearly contradicts the 'input'",
                "If the input implies certain factual info, see if the 'actual output' is consistent with that",
                "Minor omissions or ambiguities might lower the score slightly"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        )

    def measure_confidence(self, question: str, actual_output: str) -> float:
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
        )
        self.correctness_metric.measure(test_case)

        return self.correctness_metric.score
    
    def check_and_abstain(self, question: str, actual_output: str):
        score = self.measure_confidence(question, actual_output)
        reason = self.correctness_metric.reason

        return actual_output, score, reason
