import json, random
import sentencepiece as spm

valid_ids = "data/processed/tok/valid.ids.jsonl"
valid_clean = "data/processed/valid.jsonl"
spm_tgt_model = "data/processed/tok/spm_tgt.model"

sp = spm.SentencePieceProcessor(model_file=spm_tgt_model)

# load one random line from valid.ids.jsonl
ids_lines = open(valid_ids, "r", encoding="utf-8").read().strip().splitlines()
clean_lines = open(valid_clean, "r", encoding="utf-8").read().strip().splitlines()
i = random.randrange(len(ids_lines))

ids_obj = json.loads(ids_lines[i])
clean_obj = json.loads(clean_lines[i])

tgt = ids_obj["tgt_ids"]
tgt_text = sp.decode(tgt[1:-1])  # remove BOS/EOS

print("REF:", clean_obj["en"])
print("SPM:", tgt_text)
