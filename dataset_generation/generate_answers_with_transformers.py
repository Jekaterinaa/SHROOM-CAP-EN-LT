import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# --- Configuration ---
MODEL_NAME = "google/gemma-2-9b-it"
OUTPUT_FILE = "../data/lt_new_qa.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95

QUESTIONS = [
    "Kelintą vietą užėmė Žalgiris 2025-2026m. Eurolygos reguliariajame sezone?",
    "Ar Lietuva praėjo į Eurovizijos finalą 2026m.?",
    "Kas šiuo metu sudaro Lietuvos seimo valdančiąją koaliciją?",
    "Kas laimėjo Oskarą už geriausią filmą 2026m.?",
    "Kada įvyko \"Ir Velnias dėvi Pradą\" filmo premjera Lietuvoje?",
    "Kokia valiuta naudojama Lietuvoje šiuo metu?",
    "Kiek gyventojų gyvena Vilniuje 2026m. pradžioje?",
    "Ar Vilniuje yra metro?",
    "Kada Lietuva įstojo į Europos Sąjungą?",
    "Kas yra dabartinis Lietuvos prezidentas?",
    "Kokia yra Lietuvos valstybinė kalba?",
    "Kurių pilietybių imigrantų Lietuvoje yra daugiausia?",
    "Kokia yra Lietuvos nacionalinė sporto šaka?",
    "Kokia yra minimali mėnesinė alga Lietuvoje?",
    "Kiek universitetų yra Lietuvoje?",
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


def generate_answer(tokenizer, model, question: str) -> tuple[str, list[str], list[float]]:
    prompt_text = f"Klausimas: {question}\nAtsakymas:"
    combined_content = prompt_text
    messages = [
        {"role": "user", "content": combined_content},
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
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    output_tokens = []
    output_logits = []
    for i, token_id in enumerate(generated_ids):
        token_id_item = token_id.item()
        token_str = tokenizer.decode([token_id_item])
        # scores[i] has shape (1, vocab_size) — logits at step i
        token_logit = outputs.scores[i][0, token_id_item].item()
        output_tokens.append(token_str)
        output_logits.append(round(token_logit, 10))

    return answer_text, output_tokens, output_logits


def main():
    tokenizer, model = load_model(MODEL_NAME)

    results = []
    for idx, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{idx}/{len(QUESTIONS)}] Q: {question}")

        try:
            answer, output_tokens, output_logits = generate_answer(tokenizer, model, question)
            print(f"  A: {answer[:100]}...")
            print(
                f"  Tokens: {len(output_tokens)}, "
                f"Logits range: [{min(output_logits):.2f}, {max(output_logits):.2f}]"
            )

            result = {
                "index": f"custom-{idx}",
                "question": question,
                "output_text": answer,
                "output_tokens": output_tokens,
                "output_logits": output_logits,
                "model_id": MODEL_NAME,
            }
            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()