import os
import sys
import json
import argparse

# Run 'deepeval set-ollama mistral' in terminal for using Ollama
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from codes.EnhancedAgent import EnhancedAgent

def load_generated_questions(questions_path):
    with open(questions_path, "r") as f:
        data = json.load(f)
    # Combine answerable and unanswerable questions into one list.
    answerable = data.get("answerable", [])
    unanswerable = data.get("unanswerable", [])
    return answerable + unanswerable

def run_chatbot(questions_path, results_path, backend="ollama", threshold=0.7):
    questions = load_generated_questions(questions_path)
    
    # Select the appropriate secrets file based on backend.
    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets_ollama.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)

    #os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
    os.environ["PINECONE_API_KEY"] = secrets["pinecone_api_key"]
    
    # Initialize your Q&A system's agent.
    agent = EnhancedAgent(
        secrets=secrets,
        backend="ollama",
        llm_model_name="mistral",
        embed_model_name="nomic-embed-text",
        pine_index_name="td-bank-docs-new"
    )
    
    results = []
    for question in questions:
        conversation_history = ""
        answer, confidence_score, reason = agent.invoke(question, conversation_history)
        abstained = confidence_score > threshold
        results.append({
            "Question": question,
            "Generated Answer": answer,
            "Confidence Score": confidence_score,
            "Reason": reason,
            "abstained": abstained
        })
        print(f"Processed: {question[:50]}... Abstained: {abstained}")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Chatbot results saved to {results_path}")

def main():
    parser = argparse.ArgumentParser(description="Run the chatbot on generated test questions.")
    parser.add_argument("--questions_path", type=str, default="generated_questions.json", help="Path to generated questions JSON file.")
    parser.add_argument("--results_path", type=str, default="chatbot_results.json", help="Path to save chatbot results.")
    parser.add_argument("--backend", type=str, choices=["openai", "ollama"], default="ollama", help="Backend to use (selects corresponding secrets file).")
    parser.add_argument("--threshold", type=float, default=0.95, help="Confidence threshold for abstention.")
    args = parser.parse_args()
    
    run_chatbot(args.questions_path, args.results_path, args.backend, args.threshold)

if __name__ == "__main__":
    main()