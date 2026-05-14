import json
from transformers import AutoTokenizer
from pathlib import Path

MODEL_NAME = "google/gemma-2-9b-it"
OUTPUT_FILE = "../data/lt_new_answers_tokenized.jsonl"

QA_PAIRS = [
    {
        "question": "Kelintą vietą užėmė Žalgiris 2025-2026m. Eurolygos reguliariajame sezone?",
        "answer": "Žalgiris užėmė 5 vietą 2025-2026m. Eurolygos reguliariajame sezone.",
    },
    {
        "question": "Ar Lietuva praėjo į Eurovizijos finalą 2026m.?",
        "answer": "Taip, Lietuvos atstovas Lion Ceccah pateko į Eurovizijos finalą 2026m.",
    },
    {
        "question": "Kas laimėjo Oskarą už geriausią filmą 2026m.?",
        "answer": "Geriausio filmo apdovanojimą pelnė filmas \"One Battle After Another\" (\"Viena kova po kitos\") 2026m. Oskaruose.",
    },
    {
        "question": "Kada įvyko \"Ir Velnias dėvi Pradą 2\" filmo premjera Lietuvoje?",
        "answer": "Filmo \"Ir Velnias dėvi Pradą 2\" premjera Lietuvoje įvyko 2026m. gegužės 1 d.",
    },
    {
        "question": "Kiek gyventojų gyvena Vilniuje 2026m. pradžioje?",
        "answer": "2026m. sausio 3d. duomenimis  Vilniuje buvo registruoti 642,450 asmenys.",
    },
    {
        "question": "Kada prasidėjo JAV ir Izraelio karas prieš Iraną?",
        "answer": "JAV ir Izraelio karas su Iranu prasdijėo 2026m. vasario 28d.",
    },
    {
        "question": "Kas yra dabartinis JAV prezidentas?",
        "answer": "2026m. gegužės mėnesį JAV prezindentas yra Donaldas Trumpas.",
    },
    {
        "question": "Kokia yra minimali mėnesinė alga Lietuvoje 2026 metais?",
        "answer": "2026 m. Lietuvoje yra nustatyta 1153 eurų minimalią mėnesinę alga.",
    },
    {
        "question": "Ar šiuo metu Vilniuje yra metro?",
        "answer": "2026m. duomenimis  Vilniuje nėra metro.",
    },
    {
        "question": "Kurių pilietybių imigrantų Lietuvoje yra daugiausia",
        "answer": "2026m. duomenimis  Lietuvoje yra daugiausia imigrantų iš Ukrainos.",
    }
]


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = []
    for idx, pair in enumerate(QA_PAIRS, start=1):
        question = pair["question"]
        answer = pair["answer"]

        token_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)

        output_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        result = {
            "index": f"custom-{idx}",
            "question": question,
            "output_text": answer,
            "output_tokens": output_tokens,
            "model_id": MODEL_NAME,
        }
        results.append(result)
        print(f"[{idx}/{len(QA_PAIRS)}] Q: {question[:60]}  | Tokens: {len(output_tokens)}")

    # Save
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(results)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()