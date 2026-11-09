"""Generate NER JSON with a (QLoRA-adapted) LLM; output scores directly with eval/ner_f1.py.

Writes {id, output} jsonl (raw generations; ner_f1 parses + aligns strings back to char spans). Resumable:
ids already in --out are skipped. No --adapter = zero-shot baseline with the same prompt.

Kaggle (T4 x2): !pip install -q vllm
  !python /kaggle/input/<ds>/infer_vllm.py --adapter <user>/nyaya-ner-qwen7b-r16 --out /kaggle/working/preds_r16.jsonl
  !python /kaggle/input/<ds>/infer_vllm.py --out /kaggle/working/preds_zeroshot.jsonl
Fallback without vLLM: --backend hf (4-bit + PEFT, batched generate; slower).

Score: python -m eval.ner_f1 data/instruct/test.jsonl preds_r16.jsonl

Corpus run (our own judgments, inputs from ner/corpus_inputs.py, rows already without an answer):
  !python /kaggle/input/<ds>/infer_vllm.py --adapter <user>/nyaya-ner-qwen7b-r16 \
      --input /kaggle/input/<ds>/appearances.jsonl --out /kaggle/working/ner_appearances.jsonl
Generations are written every --chunk rows, so a dropped session resumes where it stopped.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ner.train_qlora import find_data, hf_token, prompt_text, read_jsonl
except ImportError:  # Kaggle: both scripts side by side in one dataset folder
    from train_qlora import find_data, hf_token, prompt_text, read_jsonl


_VLLM = {}


def run_vllm(args, prompts: list[str], token) -> list[str]:
    if "llm" not in _VLLM:
        _VLLM["llm"], _VLLM["lora"] = load_vllm(args, token)
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outs = _VLLM["llm"].generate(prompts, sp, lora_request=_VLLM["lora"])
    return [o.outputs[0].text for o in outs]


def load_vllm(args, token):
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    # every prompt shares the long instruction prefix: prefix caching skips recomputing it
    kw = dict(model=args.model, tensor_parallel_size=args.tp, dtype="half", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=True)
    lora = None
    if args.adapter:
        from huggingface_hub import snapshot_download
        path = args.adapter if os.path.isdir(args.adapter) else snapshot_download(args.adapter, token=token)
        kw.update(enable_lora=True, max_lora_rank=64)
        lora = LoRARequest("ner", 1, path)
    return LLM(**kw), lora


def run_hf(args, prompts: list[str], token) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=token, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(args.model, token=token)
    else:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, token=token, quantization_config=bnb, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, token=token)
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
    out = [None] * len(prompts)
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen):
            out[i] = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(s + args.batch, len(order))}/{len(order)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter dir or HF repo id; omit for zero-shot")
    ap.add_argument("--data", help="dir with test.jsonl (default: auto-find under /kaggle/input)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--input", help="jsonl of rows whose messages end with the user turn (ner/corpus_inputs.py)")
    ap.add_argument("--chunk", type=int, default=2000, help="rows generated between appends to --out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--tp", type=int, default=2, help="vLLM tensor parallel size (T4 x2 -> 2)")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=8, help="hf backend batch size")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no_4bit", action="store_true")
    args, _ = ap.parse_known_args()
    token = hf_token()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    recs = read_jsonl(args.input or f"{find_data(args.data)}/{args.split}.jsonl")[:args.limit]
    done = {json.loads(l)["id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} rows, {len(done)} cached, generating {len(todo)}")
    if not todo:
        return
    run = run_vllm if args.backend == "vllm" else run_hf
    for s in range(0, len(todo), args.chunk):
        batch = todo[s:s + args.chunk]
        msgs = [r["messages"] if r["messages"][-1]["role"] == "user" else r["messages"][:-1] for r in batch]
        t0 = time.time()
        texts = run(args, [prompt_text(tok, m, args.model) for m in msgs], token)
        with open(args.out, "a") as f:
            for r, t in zip(batch, texts):
                f.write(json.dumps({"id": r["id"], "output": t}, ensure_ascii=False) + "\n")
        print(f"  {s + len(batch)}/{len(todo)} written, {time.time() - t0:.0f}s for this chunk", flush=True)
    print("->", args.out)


if __name__ == "__main__":
    main()
