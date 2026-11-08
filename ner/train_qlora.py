"""QLoRA SFT of an instruct LLM on InLegalNER instruction pairs (data/instruct from ner/to_instruct.py).

Kaggle (GPU T4 x2, Internet on, secret HF_TOKEN with write access):
  Add data/instruct/{train,dev,test}.jsonl + this file + infer_vllm.py as a Kaggle Dataset, then
  !pip install -q "transformers>=4.46" peft bitsandbytes accelerate datasets
  # one run per GPU, so two ranks train in parallel. Kaggle's kernel rejects `!cmd &`
  # ("Background processes not supported"), so launch with subprocess.Popen from a Python cell:
  #   procs = [subprocess.Popen(["python", f"{DS}/train_qlora.py", "--r", str(r), "--hub_repo", REPO],
  #            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}, stdout=open(f"r{r}.log", "w"),
  #            stderr=subprocess.STDOUT) for g, r in [(0, 8), (1, 16)]]
  # then keep a cell waiting on procs, or Save & Run All ends the session and kills them.

4-bit NF4 base (double quant, fp16 compute: T4 has no bf16), LoRA on all linear projections, alpha = 2r.
Loss only on the assistant JSON (prompt tokens masked). Adapter -> <out>/adapter, pushed to <hub_repo>-r<r>.

Local smoke test (no CUDA / bitsandbytes):
  python -m ner.train_qlora --model Qwen/Qwen2.5-0.5B-Instruct --no_4bit --limit 16 --max_steps 2 --out data/qlora_smoke
"""
import argparse
import glob
import inspect
import json
import os
import time

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

IGNORE = -100
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def find_data(path: str | None) -> str:
    if path:
        return path
    hits = glob.glob("/kaggle/input/**/train.jsonl", recursive=True)
    if not hits:
        raise SystemExit("pass --data DIR containing train/dev/test.jsonl (from ner/to_instruct.py)")
    return os.path.dirname(hits[0])


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def chat_kwargs(model: str) -> dict:
    return {"enable_thinking": False} if "qwen3" in model.lower() else {}


def prompt_text(tok, messages: list[dict], model: str) -> str:
    """System + user turns rendered with the generation prompt; shared with infer_vllm.py."""
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **chat_kwargs(model))


def encode(tok, rec: dict, model: str, max_len: int) -> dict | None:
    prompt = prompt_text(tok, rec["messages"][:-1], model)
    full = tok.apply_chat_template(rec["messages"], tokenize=False, **chat_kwargs(model))
    if not full.startswith(prompt):
        raise ValueError("chat template: full conversation does not extend the generation prompt")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if ids[:len(p_ids)] != p_ids:
        raise ValueError("prompt tokens differ inside the full conversation; masking would be misaligned")
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + ids[len(p_ids):]}


def collate(pad_id: int):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), IGNORE)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


def length_grouping() -> dict:
    """Batch similar lengths (less padding). transformers 4.x: group_by_length; 5.x: train_sampling_strategy."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in params:
        return {"train_sampling_strategy": "group_by_length"}
    return {"group_by_length": True} if "group_by_length" in params else {}


class TimeLimit(TrainerCallback):
    """Stop cleanly (saving the best adapter) before Kaggle's session limit."""

    def __init__(self, hours: float):
        self.deadline = time.time() + hours * 3600

    def on_step_end(self, args, state, control, **kw):
        if time.time() > self.deadline:
            control.should_training_stop = True


def hf_token() -> str | None:
    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data")
    ap.add_argument("--out", default="/kaggle/working/qlora")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank (spec sweep: 8, 16, 64)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1, help="T4: >1 OOMs on 2k-token rows (fp32 logits x 152k vocab)")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dev_eval", type=int, default=400, help="dev rows for eval loss")
    ap.add_argument("--max_hours", type=float, default=10.5)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, help="train rows (smoke test)")
    ap.add_argument("--no_4bit", action="store_true", help="full-precision base (no CUDA / bitsandbytes)")
    ap.add_argument("--hub_repo", default=os.environ.get("HUB_REPO"))
    ap.add_argument("--seed", type=int, default=42)
    args, _ = ap.parse_known_args()
    set_seed(args.seed)
    out = f"{args.out}/r{args.r}"
    os.makedirs(out, exist_ok=True)
    token = hf_token()

    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = find_data(args.data)
    train_recs = read_jsonl(f"{data}/train.jsonl")[:args.limit]
    dev_recs = read_jsonl(f"{data}/dev.jsonl")[:args.dev_eval]
    train = [e for e in (encode(tok, r, args.model, args.max_len) for r in train_recs) if e]
    dev = [e for e in (encode(tok, r, args.model, args.max_len) for r in dev_recs) if e]
    print(f"train {len(train)} (skipped {len(train_recs) - len(train)} > {args.max_len} tokens)  dev {len(dev)}  "
          f"train tokens {sum(len(e['input_ids']) for e in train):,}")

    from peft import LoraConfig, get_peft_model
    cuda = torch.cuda.is_available()
    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(args.model, token=token, torch_dtype=torch.float32)
    else:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, token=token, quantization_config=bnb,
                                                     device_map={"": 0}, torch_dtype=torch.float16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=f"{out}/ckpt", num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * len(train) / (args.batch * args.grad_accum))),
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, fp16=cuda and not args.no_4bit,
        optim="paged_adamw_8bit" if cuda and not args.no_4bit else "adamw_torch",
        gradient_checkpointing=not args.no_4bit, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=10, report_to="none", seed=args.seed, remove_unused_columns=False, **length_grouping())
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(train),
                      eval_dataset=Dataset.from_list(dev), data_collator=collate(tok.pad_token_id),
                      callbacks=[TimeLimit(args.max_hours)])
    t0 = time.time()
    trainer.train()
    model.save_pretrained(f"{out}/adapter")
    tok.save_pretrained(f"{out}/adapter")
    log = {"args": vars(args), "best_eval_loss": trainer.state.best_metric, "steps": trainer.state.global_step,
           "hours": round((time.time() - t0) / 3600, 2), "train_examples": len(train)}
    with open(f"{out}/train_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(log)

    if args.hub_repo:
        from huggingface_hub import HfApi
        repo = f"{args.hub_repo}-r{args.r}"
        api = HfApi(token=token)
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=f"{out}/adapter")
        api.upload_file(path_or_fileobj=f"{out}/train_log.json", path_in_repo="train_log.json", repo_id=repo)
        print("pushed to", repo)


if __name__ == "__main__":
    main()
