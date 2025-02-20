from trulens_eval import TruLlama, TruChain, Feedback
from trulens_eval import OpenAI as fOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
import warnings
import numpy as np
import pandas as pd

class QuestionLoader:
    def __init__(self, filename):
        self.filename = filename
        self.eval_questions = []

    def load_questions(self):
        with open(self.filename, 'r') as file:
            for line in file:
                self.eval_questions.append(line.strip())
        return self.eval_questions

class Evaluator:
    def __init__(self):
        self.tru = Tru()
        self.tru.reset_database()
        self.feedbacks = self._setup_feedbacks()

    def _setup_feedbacks(self):
        provider = fOpenAI()
        qa_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance").on_input_output()
        groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness").on_input_output()
        context_relevance = Feedback(provider.qs_relevance_with_cot_reason, name = "Context Relevance").on_input_output()
        return [qa_relevance, groundedness, context_relevance]

    def get_feedbacks(self):
        return self.feedbacks

    def get_records_and_feedback(self, app_ids=[]):
        return self.tru.get_records_and_feedback(app_ids=app_ids)

    def get_leaderboard(self, app_ids=[]):
        return self.tru.get_leaderboard(app_ids=app_ids)

    def run_dashboard(self):
        self.tru.run_dashboard()

class RagChain:
    def __init__(self, retriever, llm, feedbacks, app_id='custom_rag'):
        self.retriever = retriever
        self.llm = llm
        self.app_id = app_id
        self.feedbacks = feedbacks
        self.rag_chain = self._setup_rag_chain()
        self.tru_recorder = self._initialize_tru_chain()

    def _setup_rag_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = {
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        } | prompt | self.llm | StrOutputParser()
        return rag_chain

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _initialize_tru_chain(self):
        return TruChain(self.rag_chain, app_id=self.app_id, feedbacks=self.feedbacks)

    def get_recorder(self):
        return self.tru_recorder

class Dashboard:
    def __init__(self, tru_evaluator):
        self.tru_evaluator = tru_evaluator
        pd.set_option("display.max_colwidth", None)

    def display_records(self, app_ids=[]):
        records, feedback = self.tru_evaluator.get_records_and_feedback(app_ids=app_ids)
        print(records[["input", "output"] + feedback])

    def show_leaderboard(self, app_ids=[]):
        leaderboard = self.tru_evaluator.get_leaderboard(app_ids=app_ids)
        print(leaderboard)

    def run(self):
        self.tru_evaluator.run_dashboard()

class EvaluationRunner:
    def __init__(self, questions, tru_recorder, llm):
        self.questions = questions
        self.tru_recorder = tru_recorder
        self.llm = llm

    def run(self):
        for question in self.questions:
            with self.tru_recorder as recording:
                self.llm.invoke(question)
