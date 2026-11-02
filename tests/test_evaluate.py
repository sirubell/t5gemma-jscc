import json

import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from jscc import evaluate
from jscc.evaluate import CocoScorer, clean_caption
from tests.helpers import tiny_model


def tokenizer(newline):
    """Word-level tokenizer over the tiny vocabulary; id ``newline`` is the newline token.
    ("abcdef" is the probe word transformers' stop-string matcher tokenizes.)"""
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}." if i != 28 else "abcdef": i for i in range(3, 32)}}
    vocab = {("\n" if i == newline else t): i for t, i in vocab.items()}
    backend = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")


class Image:
    def __init__(self, seed):
        self.pixels = torch.randn(3, 8, 8, generator=torch.Generator().manual_seed(seed))

    def convert(self, mode):
        return self


class Processor:
    """Stands in for the Gemma3 processor: per-image pixels, one [BOI, 4 image tokens, EOI, word] block per image."""

    def __init__(self, tok):
        self.tokenizer, self.images = tok, 0

    def __call__(self, images, text, **kwargs):
        self.images += sum(map(len, images))
        ids = torch.tensor([[29, 31, 31, 31, 31, 30, 5 + j] for j in range(len(images[0]))]).flatten()
        return {"input_ids": ids.repeat(len(images), 1), "attention_mask": torch.ones(len(images), ids.numel(), dtype=torch.long),
                "pixel_values": torch.stack([i.pixels for item in images for i in item])}


class Data:
    ids = {"report_ids": list(range(1, 8))}
    demo_images = [Image(-j) for j in range(1, 5)]
    demo_captions = ["a", "b", "c", "d"]

    def row(self, i):
        return {"image": Image(i), "answer": ["w3. w4."], "file_name": f"COCO_val2014_{i:012d}.jpg"}


def panels(scorer, model, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "cider", lambda refs, caps: (0.0, [float(len(caps[k][0]["caption"])) for k in refs]))
    out = []
    for site in ["enc_emb", "enc_l1", "enc_fn", "dec_l0", "dec_l1"]:
        model.set_site(site)
        for condition in ["no_noise", "-6"]:
            torch.manual_seed(0)
            scorer(model, condition, tmp_path)
            out.append((tmp_path / f"captions_{site}_{condition}.jsonl").read_text())
    return out


def reference(model, processor, config, data, tmp_path, monkeypatch):
    """The previous scorer loop: processor per batch and panel, generation to EOS or max_new_tokens."""
    def call(self, model, condition, output):
        e = self.config["evaluation"]
        lines = []
        with model.transmission(evaluate._snr(condition)):
            for start in range(0, len(self.rows), e["batch_size"]):
                batch = self.rows[start:start + e["batch_size"]]
                inputs = self.processor(images=[self.data.demo_images + [r["image"].convert("RGB")] for r in batch],
                                        text=[self.prompt] * len(batch))
                generated = model.generate(**inputs, max_new_tokens=e["max_new_tokens"], do_sample=False, num_beams=1,
                                           use_cache=True)
                lines += [{"image_id": evaluate.image_id(r), "caption": clean_caption(
                    self.processor.tokenizer.decode(t, skip_special_tokens=True))} for r, t in zip(batch, generated)]
        output.mkdir(parents=True, exist_ok=True)
        with (output / f"captions_{model.site}_{condition}.jsonl").open("w") as f:
            for line in lines:
                f.write(json.dumps({**line, "cider": float(len(line["caption"]))}) + "\n")
    monkeypatch.setattr(CocoScorer, "__call__", call)
    result = panels(CocoScorer(config, processor, data), model, tmp_path, monkeypatch)
    monkeypatch.undo()
    return result


def test_coco_cached_inputs_and_newline_stop_keep_captions(tmp_path, monkeypatch):
    model = tiny_model(layers=3).eval()
    config = {"evaluation": {"batch_size": 3, "max_new_tokens": 12}}
    tok = tokenizer(25)  # the tiny model emits token 25 both first and mid-caption (e.g. dec_l0: 4 3 25 ...)
    old = reference(model, Processor(tok), config, Data(), tmp_path / "old", monkeypatch)
    processor = Processor(tok)
    new = panels(CocoScorer(config, processor, Data()), model, tmp_path / "new", monkeypatch)
    assert new == old
    assert processor.images == 7 * 5  # each report batch preprocessed once for all 10 panels
    assert {json.loads(line)["caption"] for panel in new for line in panel.splitlines()} > {""}
    inputs = processor(images=[Data.demo_images + [Image(i)] for i in range(1, 4)], text=None)
    model.set_site("dec_l0")
    with model.transmission(None):
        full = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        stopped = model.generate(**inputs, max_new_tokens=12, do_sample=False, stop_strings="\n", tokenizer=tok)
    assert stopped.shape[1] == 4 < full.shape[1] and torch.equal(stopped, full[:, :4])  # BOS, 4, 3, newline
