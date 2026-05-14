import json

input_path = "../data/lt_train_data.jsonl"
output_path = "../data/lt_train_label.jsonl"

with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        label = {
            "index": entry["index"],
            "has_fluency_mistakes": "",
            "has_factual_mistakes": ""
        }
        f_out.write(json.dumps(label, ensure_ascii=False) + "\n")

print(f"Labels file written to {output_path}")
