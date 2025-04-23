import os
import sys
import json
import argparse
import csv

# Run 'deepeval set-ollama mistral' in terminal for using Ollama
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from codes.EnhancedAgent import EnhancedAgent

def load_generated_questions(questions_path):
    with open(questions_path, "r", encoding="utf-8") as f:
        question_objects = json.load(f)
    return question_objects  # List of dicts with 'Question' and 'Answer'

def run_chatbot(questions_path, results_path, backend="ollama", threshold=0.7, csv_path=None):
    question_objects = load_generated_questions(questions_path)

    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"secrets_{backend}.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)

    os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
    os.environ["PINECONE_API_KEY"] = secrets["pinecone_api_key"]

    agent = EnhancedAgent(
        secrets=secrets,
        trace_on=False,
        backend=backend,
        llm_model_name="mistral" if backend == "ollama" else "gpt-4o",
        embed_model_name="nomic-embed-text" if backend == "ollama" else "text-embedding-ada-002",
        pine_index_name="td-bank-docs-new" if backend == "ollama" else "td-bank-docs-openai",
        use_cohere=True,
        hybrid_alpha=0.5,
        confidence_threshold=threshold
    )

    results = []
    csv_rows = []

    for item in question_objects:
        question = item.get("Question")
        expected_answer = item.get("Answer")
        conversation_history = ""
        answer, top_context_vectors, confidence_score, reason = agent.invoke(question, conversation_history)
        abstained = confidence_score < threshold
        results.append({
            "Question": question,
            "Expected Answer": expected_answer,
            "Generated Answer": answer,
            "Confidence Score": confidence_score,
            "Reason": reason,
            "Abstained": abstained
        })
        csv_rows.append({
            "Question": question,
            "Confidence Score": confidence_score,
            "Abstained": 1 if abstained else 0
        })
        print(f"Processed: {question[:50]}\n{answer}\n{reason}\n... Conf={confidence_score} Abstained: {abstained}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Chatbot results saved to {results_path}")

    if csv_path:
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Question", "Confidence Score", "Correct"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Confidence data saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Run the chatbot on generated test questions.")
    parser.add_argument("--questions_path", type=str, default="answerable_questions.json", help="Path to generated questions JSON file.")
    parser.add_argument("--results_path", type=str, default="answerable_results.json", help="Path to save chatbot results.")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional: path to save confidence scores and correctness as CSV.")
    parser.add_argument("--backend", type=str, choices=["openai", "ollama"], default="ollama", help="Backend to use (selects corresponding secrets file).")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for abstention.")
    args = parser.parse_args()

    run_chatbot(args.questions_path, args.results_path, args.backend, args.threshold, args.csv_path)

if __name__ == "__main__":
    main()
