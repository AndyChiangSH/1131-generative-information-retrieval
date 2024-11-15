import os
import json
import openai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


# Load claims data
def load_claims(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Load prompt
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# Predict label using OpenAI API
def predict_label(prompt):
    if CONFIG["api"] == "sambanova":
        client = openai.OpenAI(
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )
    elif CONFIG["api"] == "openai":
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    response = client.chat.completions.create(
        model=CONFIG['llm'],
        messages=[
            {"role": "system", "content": "You are a fact-checker."}, 
            {"role": "user", "content": prompt}
        ],
        max_tokens=1,
        temperature=0.1,
        top_p=0.1,
    )
    
    label = response.choices[0].message.content
    
    # print("label:", label)
    
    return int(label)


if __name__ == "__main__":
    print("> Start llm!")
    
    # Define configuration
    print("> Define configuration...")
    CONFIG = {
        "input_path": "retriever/TF-IDF/top-10-unique/",
        "output_path": "submission/",
        "api": "openai",
        "llm": "gpt-4o-mini",
        "in_context_learning": "zero-shot"
    }
    print("CONFIG:", CONFIG)
    
    # Load environment variables
    load_dotenv()

    # Load test claims
    print("> Load test claims...")
    test_claims = load_claims(os.path.join(CONFIG["input_path"], "test.json"))
    
    # Prepare submission list
    submission_data = []

    # Predict labels for the test data
    print("> Predict labels for the test data...")
    for claim_obj in tqdm(test_claims):
        metadata = claim_obj.get("metadata", {})
        claim = metadata.get("claim", "")
        claim_id = metadata.get("id", "")
        evidence_sentences = metadata.get("top_relevant_sentences", [])
        
        # Load prompt
        prompt = load_prompt(f"classifier/prompt/{CONFIG['in_context_learning']}.txt")
        prompt = prompt.replace("{claim}", claim)
        evidence_text = "\n".join(evidence_sentences)
        prompt = prompt.replace("{evidence_sentences}", evidence_text)
        
        # print("prompt:", prompt)
        
        # Use LLM to predict the label
        label = predict_label(prompt)
        
        # Append prediction to the submission list
        submission_data.append({"id": int(claim_id), "rating": int(label)})

    # Create a DataFrame and save to CSV
    print("> Create a DataFrame and save to CSV...")
    submission_df = pd.DataFrame(submission_data)
    submission_path = f"{CONFIG['output_path']}/{CONFIG['llm']}_{CONFIG['in_context_learning']}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission file saved to: {submission_path}")
    
    print("> End llm!")
