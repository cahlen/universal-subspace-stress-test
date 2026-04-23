"""
Download N LoRAs from Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task*.

This is the same rank-16 Mistral-7B LoRA collection the paper cites
(Brüel-Gabrielsson et al. 2024, trained on Natural Instructions v2 tasks).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

N = int(os.environ.get("N_LORAS", "100"))
OUT = Path(os.environ.get("LORA_DIR", "./data/loras"))
OUT.mkdir(parents=True, exist_ok=True)

api = HfApi()
repos = [
    m.modelId for m in api.list_models(author="Lots-of-LoRAs", limit=1000)
    if "Mistral-7B-Instruct-v0.2-4b-r16" in m.modelId
]
repos = sorted(repos)[:N]
print(f"[plan] will download {len(repos)} LoRAs into {OUT}", flush=True)


def fetch(repo: str):
    safe_name = repo.replace("/", "__")
    dest = OUT / safe_name
    if (dest / "adapter_model.safetensors").exists():
        return repo, "cached"
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    return repo, "downloaded"


failures = []
done = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(fetch, r): r for r in repos}
    for fut in as_completed(futs):
        r = futs[fut]
        try:
            repo, status = fut.result()
            done += 1
            if done % 10 == 0 or status == "cached":
                print(f"  [{done}/{len(repos)}] {status}: {repo}", flush=True)
        except Exception as e:
            failures.append((r, str(e)))
            print(f"  FAIL {r}: {e}", flush=True)

print(f"[done] {done} ok, {len(failures)} failed")
if failures:
    for r, e in failures[:5]:
        print("  ", r, "::", e)
