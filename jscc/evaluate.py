"""Task scoring per site and channel condition.

Conditions: ``vanilla`` (codec and channel bypassed: the unsplit model),
``no_noise`` (codec + power normalization, no noise), or an SNR in dB.
HellaSwag: lm-eval five-shot ``acc_norm`` on the official validation split.
COCO: greedy captions on the report images, PTB tokenizer + CIDEr.
Each (site, condition) panel reseeds the noise with ``evaluation.noise_seed``.
"""
import copy
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .tasks import caption_prompt, image_id


def _seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _snr(condition):
    return None if condition in ("no_noise", "vanilla") else float(condition)


# ---- HellaSwag --------------------------------------------------------
class HellaSwagScorer:
    def __init__(self, config, tokenizer):
        from lm_eval.tasks import TaskManager
        from lm_eval.api.task import ConfigurableTask
        from lm_eval.tasks._yaml_loader import load_yaml
        self.config, self.tokenizer = config, tokenizer
        self.manager = TaskManager()
        yaml_path = self.manager.task_index["hellaswag"].yaml_path
        assert yaml_path is not None
        task_config = copy.deepcopy(load_yaml(yaml_path, resolve_func=True))
        task_config["dataset_path"] = config["data"]["name"]
        task_config.setdefault("dataset_kwargs", {})["revision"] = config["data"]["revision"]
        self.task = ConfigurableTask(config=task_config)

    def __call__(self, model, condition, output):
        from lm_eval.api.registry import get_model
        from lm_eval.evaluator import simple_evaluate
        e = self.config["evaluation"]
        dtype = next(model.base.parameters()).dtype
        harness: Any = get_model("hf")
        adapter = harness(pretrained=model.base, tokenizer=self.tokenizer, backend="seq2seq",
                                  batch_size=e["batch_size"], softmax_dtype=torch.float32,
                                  mixed_precision_dtype=dtype if dtype in (torch.bfloat16, torch.float16) else None)
        with model.transmission(_snr(condition), bypass=condition == "vanilla"):
            result = simple_evaluate(model=adapter, tasks=[copy.deepcopy(self.task)], task_manager=self.manager,
                                     num_fewshot=5, limit=e.get("items"), bootstrap_iters=0,
                                     random_seed=0, numpy_random_seed=1234, torch_random_seed=0,
                                     fewshot_random_seed=1234)
        if result is None:
            raise RuntimeError("lm-eval returned no results")
        out: Any = result
        m = out["results"]["hellaswag"]
        return {"acc_norm": m["acc_norm,none"], "acc": m["acc,none"],
                "items": out.get("n-samples", {}).get("hellaswag", {}).get("effective"), "channel_uses": dict(model.uses)}


# ---- COCO ---------------------------------------------------------------
def clean_caption(text):
    text = text.split("\n")[0].split("\r")[0].strip()
    return text.split(". ")[0] + "." if ". " in text else text


def cider(references, captions):
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tok = PTBTokenizer()
    refs, caps = tok.tokenize(references), tok.tokenize(captions)
    refs = {k: refs[k] for k in references}
    caps = {k: caps[k] for k in references}
    score, per_image = Cider().compute_score(refs, caps)
    return float(score), [float(s) for s in per_image]


class CocoScorer:
    def __init__(self, config, processor, data):
        self.config, self.processor, self.data = config, processor, data
        ids = data.ids["report_ids"][: config["evaluation"].get("items") or None]
        self.rows = [data.row(i) for i in ids]
        self.prompt = caption_prompt(data.demo_captions)

    @torch.no_grad()
    def __call__(self, model, condition, output):
        e = self.config["evaluation"]
        param = next(model.base.parameters())
        refs, caps, lines = {}, {}, []
        with model.transmission(_snr(condition), bypass=condition == "vanilla"):
            for start in tqdm(range(0, len(self.rows), e["batch_size"]), desc=f"{model.site}:{condition}"):
                batch = self.rows[start:start + e["batch_size"]]
                inputs = self.processor(images=[self.data.demo_images + [r["image"].convert("RGB")] for r in batch],
                                        text=[self.prompt] * len(batch), return_tensors="pt", padding=True,
                                        truncation=False)
                kwargs = {k: inputs[k].to(param.device, dtype=param.dtype if k == "pixel_values" else None)
                          for k in ("input_ids", "attention_mask", "pixel_values")}
                generated = model.generate(**kwargs, max_new_tokens=e["max_new_tokens"], do_sample=False,
                                           num_beams=1, use_cache=True)
                for row, tokens in zip(batch, generated):
                    key = image_id(row)
                    caption = clean_caption(self.processor.tokenizer.decode(tokens, skip_special_tokens=True))
                    refs[key] = [{"caption": t.lower().strip()} for t in row["answer"]]
                    caps[key] = [{"caption": caption.lower().strip()}]
                    lines.append({"image_id": key, "caption": caption})
        score, per_image = cider(refs, caps)
        output.mkdir(parents=True, exist_ok=True)
        with (output / f"captions_{model.site}_{condition}.jsonl").open("w") as f:
            for line, s in zip(lines, per_image):
                f.write(json.dumps({**line, "cider": s}) + "\n")
        return {"cider": score, "items": len(lines), "channel_uses": dict(model.uses)}


# ---- driver -------------------------------------------------------------
def evaluate(model, processor, config, sites, output: Path, data=None):
    """Score every (site, condition) panel; return ``{"site|condition": metrics}``.

    ``vanilla`` does not depend on the site and is scored once (key ``vanilla``).
    """
    e = config["evaluation"]
    scorer = (CocoScorer(config, processor, data) if config["task"] == "coco"
              else HellaSwagScorer(config, processor))
    model.eval()
    results = {}
    path = output / "results.json"
    if path.exists():  # resume a partially scored checkpoint
        results = json.loads(path.read_text())
    panels = [(sites[0], "vanilla")] if "vanilla" in e["conditions"] else []
    panels += [(s, str(c)) for s in sites for c in e["conditions"] if c != "vanilla"]
    for site, condition in panels:
        key = "vanilla" if condition == "vanilla" else f"{site}|{condition}"
        if key in results:
            continue
        model.set_site(site)
        _seed(e.get("noise_seed", 0))
        results[key] = scorer(model, condition, output)
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1) + "\n")
    return results
