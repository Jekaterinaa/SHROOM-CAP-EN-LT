import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from prompts.system_prompt import lt_transformers_system_prompt
from prompts.user_prompt import lt_transformers_user_prompt

# --- Configuration ---
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "google/gemma-2-9b-it"
OUTPUT_FILE = "data/lt_train_data.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
NUM_EXAMPLES = 85
BATCH_SIZE = 5

TOPICS = [
    "Lithuanian history (dates, events, historical figures)",
    "Lithuanian geography (cities, rivers, lakes, elevations)",
    "Lithuanian literature and culture (writers, books, dates)",
    "European history (dates, wars, treaties)",
    "Natural sciences (chemical elements, physics constants, biology)",
    "Mathematics (formulas, numbers, geometry)",
    "Sports (Olympic games, records, dates)",
    "Music (composers, works, dates)",
    "World geography (capitals, populations, areas)",
    "Lithuanian language (grammar rules, word origins)",
]


def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def generate_qa_pairs(tokenizer, model, topic: str, n: int) -> list[dict]:
    # messages = [
    #     {"role": "system", "content": lt_transformers_system_prompt},
    #     {"role": "user", "content": lt_transformers_user_prompt.format(n=n, topic=topic)},
    # ]
    combined_user_content = lt_transformers_system_prompt.strip() + "\n\n" + lt_transformers_user_prompt.format(n=n, topic=topic)
    messages = [
        {"role": "user", "content": combined_user_content},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Parse JSON from output
    text = raw_output.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        print(f"  WARNING: Could not find JSON array in output. Raw: {text[:300]}")
        return []

    try:
        pairs = json.loads(text[start:end + 1])
        valid = []
        for p in pairs:
            if isinstance(p, dict) and "question" in p and "answer" in p:
                q = p["question"].strip()
                a = p["answer"].strip()
                if q and a:
                    valid.append({"question": q, "answer": a})
        return valid
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}. Raw: {text[:300]}")
        return []


def get_answer_tokens_and_logits(
    tokenizer,
    model,
    question: str,
    answer: str,
) -> tuple[list[str], list[float]]:
    prompt_text = f"Klausimas: {question}\nAtsakymas:"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    answer_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_tensor = torch.tensor([[eos_id]], device=model.device)
        answer_ids = torch.cat([answer_ids, eos_tensor], dim=1)

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    with torch.no_grad():
        outputs = model(full_ids)
    logits = outputs.logits

    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    output_tokens = []
    output_logits = []

    for i in range(answer_len):
        logit_pos = prompt_len - 1 + i
        token_id = answer_ids[0, i].item()
        token_logit = logits[0, logit_pos, token_id].item()
        token_str = tokenizer.decode([token_id])
        output_tokens.append(token_str)
        output_logits.append(round(token_logit, 10))

    return output_tokens, output_logits


def main():
    tokenizer, model = load_model(MODEL_NAME)

    results = []
    examples_needed = NUM_EXAMPLES
    topic_idx = 0

    while len(results) < examples_needed:
        topic = TOPICS[topic_idx % len(TOPICS)]
        remaining = examples_needed - len(results)
        n = min(BATCH_SIZE, remaining)

        print(f"\n[{len(results)}/{examples_needed}] Generating {n} pairs for: {topic}")

        pairs = generate_qa_pairs(tokenizer, model, topic, n)

        if not pairs:
            print("  No valid pairs generated, retrying with next topic...")
            topic_idx += 1
            continue

        for pair in pairs:
            if len(results) >= examples_needed:
                break

            question = pair["question"]
            answer = pair["answer"]

            try:
                output_tokens, output_logits = get_answer_tokens_and_logits(
                    tokenizer, model, question, answer
                )

                result = {
                    "index": f"lt-train-{len(results) + 1}",
                    "question": question,
                    "output_text": answer,
                    "output_tokens": output_tokens,
                    "output_logits": output_logits,
                    "topic": topic,
                    "model_id": MODEL_NAME,
                }
                results.append(result)

                print(
                    f"  [{len(results)}/{examples_needed}] "
                    f"Q: {question[:70]}... | "
                    f"Tokens: {len(output_tokens)}, "
                    f"Logits range: [{min(output_logits):.2f}, {max(output_logits):.2f}]"
                )
            except Exception as e:
                print(f"  ERROR getting tokens/logits: {e}")
                continue

        topic_idx += 1

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(results)} Lithuanian examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()