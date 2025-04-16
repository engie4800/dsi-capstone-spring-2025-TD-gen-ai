import os
import json
from openai import OpenAI



secrets_file = os.path.join(os.path.dirname(__file__), "..", "secrets_openai.json")
with open(secrets_file, "r") as f:
    secrets = json.load(f)
    # Set the environment variable (and also assign it to openai.api_key)
os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_prompts(prompt_template, num_prompts):
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a creative assistant that generates questions based on TD Bank financial reports."},
        {"role": "user", "content": prompt_template}
    ],
    temperature=0.8,
    max_tokens=1000,
    n=1)
    prompts_text = response.choices[0].message.content
    # Split output into lines and filter out empty lines.
    prompts = [q.strip() for q in prompts_text.split("\n") if q.strip()]
    return prompts[:num_prompts]

def generate_answerable_prompts(num_questions):
    prompt_template = (
        f"Generate {num_questions} distinct answerable questions about TD Bank based on its publicly available financial reports "
        "from quarterly and annual results. The questions should be factual and directly reference details such as net income, "
        "revenue figures, branch numbers, and strategic initiatives. Use diverse tones (casual, formal, conversational, professional)."
    )
    return generate_prompts(prompt_template, num_questions)

def generate_unanswerable_prompts(num_questions):
    prompt_template = (
        f"Generate {num_questions} distinct unanswerable questions that are similar in style to questions about TD Bank's financial reports, "
        "but refer to details or future projections that are not available in the documents. The questions should mimic the tone "
        "of the answerable ones (casual, formal, conversational, professional) but ask for information that the reports do not provide."
    )
    return generate_prompts(prompt_template, num_questions)

def main():
    num_questions = 1
    answerable = generate_answerable_prompts(num_questions)
    unanswerable = generate_unanswerable_prompts(num_questions)
    all_questions = {
        "answerable": answerable,
        "unanswerable": unanswerable
    }
    output_file = "generated_questions.json"
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=4)
    print(f"Generated questions saved to {output_file}")

if __name__ == "__main__":
    main()