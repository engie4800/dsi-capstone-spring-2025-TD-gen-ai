from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

class GEvalStrict(GEval):
    """
    A stricter version of DeepEval's GEval that forces the model to output a score in the range [0.0, 1.0].
    Adds additional instruction to the prompt to prevent scoring on a 1–10 scale or other formats.
    ---04/16/2025: Doesn't always work!!!
    """

    def build_prompt(self, test_case: LLMTestCase) -> str:
        # Generate the base prompt from the parent
        base_prompt = super().build_prompt(test_case)

        # Append strict formatting instructions
        strict_instructions = (
            "\n\nIMPORTANT:\n"
            "- Return only a **valid JSON object**.\n"
            "- The JSON object must be of the format:\n"
            '  { "score": float between 0.0 and 1.0, "reason": "..." }\n'
            "- Do not use integers, percentages, or numbers greater than 1.0.\n"
            "- Do not wrap the JSON in markdown like ```json.\n"
            "- Do not include explanations outside the JSON object.\n"
        )

        return base_prompt + strict_instructions
