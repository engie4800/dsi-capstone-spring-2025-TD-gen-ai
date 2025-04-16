from .EnchancedResponseChain import EnhancedResponseChain
from .ConfidenceCheck import ConfidenceChecker
from .GenericLLM import GenericLLM


class EnhancedAgent:
    def __init__(self, secrets, trace_on, backend, llm_model_name, embed_model_name, pine_index_name,
                 use_cohere, hybrid_alpha, confidence_threshold):
        self.retriever = EnhancedResponseChain(secrets, trace_on, backend, llm_model_name, embed_model_name, pine_index_name,
                                               use_cohere, hybrid_alpha)
        self.trace_on = trace_on
        # create an LLM based on the backend environment
        llm = GenericLLM(secrets, backend, llm_model_name)
        self.confidence_checker = ConfidenceChecker(llm, threshold=confidence_threshold)
        # next line is required because we implemented a custom LLM class (GenericLLM)
        # if the next line is not set, we get None errors
        self.confidence_checker.evaluation_cost = 0

    def invoke(self, query, conversation_history="", expected_output=None):
        #return self.retriever.invoke(query, conversation_history)

        result, top_context_vectors = self.retriever.answer_question_with_citations(
            query, conversation_history
        )

        final_answer, confidence_score, reason = self.confidence_checker.check_and_abstain(
            question=query,
            actual_output=result
        )
        
        # low-confidence cases
        if confidence_score < self.confidence_checker.threshold:
            if self.trace_on:
                print(f"Abstaining! confidence_score={confidence_score} < threshold={self.confidence_checker.threshold}")
                final_answer = "I'm not confident enough in my answer. Please check the sources or ask a different question."
        
        return final_answer, top_context_vectors, confidence_score, reason
    

# Xiqian use for generate questions
# class EnhancedAgent:
#     def __init__(
#         self,
#         secrets,
#         backend="ollama",
#         llm_model_name="mistral",
#         embed_model_name="nomic-embed-text",
#         pine_index_name="td-bank-docs-new",
#         use_cohere=True
#     ):
#         self.retriever = EnhancedResponseChain(
#             secrets,
#             backend,
#             llm_model_name,
#             embed_model_name,
#             pine_index_name,
#             use_cohere
#         )
#         self.confidence_checker = ConfidenceChecker(threshold=0.7)

#     def invoke(self, query, conversation_history="", expected_output=None):
#         """
#         1. Get the answer from the retrieval pipeline.
#         2. Check the answer using a DeepEval-based confidence checker.
#         3. Return the final answer, confidence score, and reason.
#         """
#         result, top_context_vectors = self.retriever.answer_question_with_citations(
#             query, conversation_history
#         )
#         final_answer, confidence_score, reason = self.confidence_checker.check_and_abstain(
#             question=query,
#             actual_output=result
#         )
#         return final_answer, confidence_score, reason

