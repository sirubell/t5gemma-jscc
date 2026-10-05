"""Small real Transformers model shared by both task test suites."""



from jscc.config import save_config
from jscc.studies import StudyPlan, expand_study, export_study
from jscc.checkpoint_readers import (
    coco_native_SPLITS, coco_native_CONDITIONS, coco_native_PRESENTATIONS,
    coco_native_digest, coco_native_validate_measurements, coco_native_verify_bindings,
    coco_native_integer, coco_native_estimate, coco_native_numerical_limitations,
    coco_native_selected_indices, coco_native_scenarios,
)

import math
import yaml
from jscc.config import load_config
from jscc.qualification import check_reconstruction_recipe
from jscc.checkpoint_readers import check_reconstruction_plan

import gzip
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from torch.utils.data import DataLoader, Subset
from jscc import training
from jscc.config import validate_config
from jscc.presentation import tensor_digest
from jscc.gradient_diagnostics import selection_ids, validate_fixture

import copy
import random
from jscc.config import resolve_codec_configs
from jscc.models import split_model
from jscc.models.codec import FAMILIES, codec_family_context
import torch
import gc
from contextlib import nullcontext
from jscc.shared_inputs import IndexedCollator, IndexedRows
from jscc.qualification import checked_gradient_names
from transformers import T5Gemma2Config, T5Gemma2ForConditionalGeneration


def tiny_backbone(num_hidden_layers=2):
    text = {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": num_hidden_layers, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "query_pre_attn_scalar": 8, "max_position_embeddings": 64,
            "layer_types": ["full_attention"] * num_hidden_layers}
    vision = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
              "num_attention_heads": 2, "image_size": 8, "patch_size": 4}
    config = T5Gemma2Config(
        encoder={"text_config": text, "vision_config": vision, "mm_tokens_per_image": 4,
                 "boi_token_index": 29, "eoi_token_index": 30},
        decoder=text, image_token_index=31,
    )
    model = T5Gemma2ForConditionalGeneration(config)
    # HF initializes this projector to zero. A zero projector erases all pixel
    # differences and makes image-route tests meaningless; pretrained weights
    # replace it when loading the real model. Use an identity in this fixture.
    with torch.no_grad():
        model.get_encoder().multi_modal_projector.mm_input_projection_weight.copy_(torch.eye(16))
    return model


def experimental_build_model(config):
    family = config["experiment"]["codec_family"]
    if config["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("E5 factory only supports enc_l9")
    with codec_family_context(family, config["seed"]):
        return split_model.build_model(config)


def pilot_config(base, output, family):
    if family not in FAMILIES:
        raise ValueError("unknown family")
    cfg = copy.deepcopy(base)
    if cfg["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("E5 requires enc_l9")
    hidden, _ = resolve_codec_configs(cfg["codec"])
    if (hidden["hidden_dim"], hidden["bottleneck_dim"], hidden["layernorm"], hidden["snr_film"]) != (1152, 512, "none", False):
        raise ValueError("requires H1152/B512 outer none, FiLM off reference config")
    if cfg["protocol"] != "corrected-baseline-v2-native-decoder-inputs" or cfg["task"] != "hellaswag":
        raise ValueError("requires HellaSwag native-v2")
    policy = cfg["data"].get("prompt_policy", {})
    if policy.get("mode") != "five_shot" or policy.get("source_max_length") != 2048 or cfg["data"]["max_length"] != 512:
        raise ValueError("requires unchanged five-shot source2048/target512 policy")
    if cfg["seed"] != 0 or cfg["model"]["dtype"] != "bfloat16":
        raise ValueError("requires seed0/BF16 reference")
    if cfg["channel"]["type"] != "awgn" or not cfg["channel"]["train_noise"] or cfg["channel"]["train_snr_range"] != [-6, 18]:
        raise ValueError("requires production Uniform[-6,18] AWGN")
    if (cfg["training"]["kl_weight"], cfg["training"]["mse_weight"]) != (1., .1):
        raise ValueError("requires original KL + 0.1 nMSE objective")
    cfg["run"].update(name=f"evening-e5-{family}", output_dir=str(output / "runs"), wandb_project=None)
    cfg["experiment"] = {"name": "EVENING_SMALL_SUITE_V1", "codec_family": family,
                         "policy": "fresh-500-horizon-family-screen", "long_execution_authorized": False}
    cfg["training"].update(max_steps=500, schedule_steps=500, batch_size=64, gradient_accumulation=1,
        warmup_ratio=.05, lr=2e-4, log_every=50, eval_every=250, save_steps=[250, 500],
        validation_batches=0, validation_snrs=["no_noise"], patience=None, max_minutes=None,
        presentation_stream={"policy": "epoch-permutations-v1", "seed": 0, "total_presentations": 32000},
        paired_randomness={"policy": "step-microbatch-stream-v1", "seed": 0, "audit_steps": list(range(1, 501))})
    cfg["training"].pop("feature_summary", None)
    return cfg


def selection_indices(ids):
    if len(ids) < 128 or len(set(ids)) != len(ids):
        raise ValueError("selection requires at least128 unique original holdout IDs")
    positions = list(range(len(ids)))
    random.Random(20260921).shuffle(positions)
    return positions[:128]


ARMS = {"W005": .05, "W05": .5, "W5": 5.}


def screen_config(base, output, arm, preflight=False):
    cfg = copy.deepcopy(base)
    if cfg["split"] != {"stack": "dec", "where": "after_layer", "index": 20}:
        raise ValueError("screen requires dec_l20 after_layer")
    main, memory = resolve_codec_configs(cfg["codec"])
    if main["layernorm"] != "none" or memory["layernorm"] != "both":
        raise ValueError("screen requires main none / memory both")
    for codec in (main, memory):
        if (codec["hidden_dim"], codec["bottleneck_dim"], codec["n_res_blocks"], codec["activation"], codec["snr_film"]) != (1152, 512, 2, "gelu", False):
            raise ValueError("screen architecture differs from approved residual codec")
    policy = cfg["data"].get("prompt_policy", {})
    if cfg["task"] != "hellaswag" or cfg["protocol"] != "corrected-baseline-v2-native-decoder-inputs" or cfg["seed"] != 0:
        raise ValueError("requires seed0 native-v2 HellaSwag")
    if policy.get("mode") != "five_shot" or policy.get("source_max_length") != 2048 or cfg["data"]["max_length"] != 512:
        raise ValueError("requires existing five-shot source2048 target512")
    if cfg["model"]["dtype"] != "bfloat16" or cfg["channel"]["type"] != "awgn" or not cfg["channel"]["train_noise"] or cfg["channel"]["train_snr_range"] != [-6, 18]:
        raise ValueError("requires BF16 backbone and paired U[-6,18] AWGN")
    steps = 2 if preflight else 1000
    cfg["run"].update(name=f"main-weight-{arm}-{'preflight' if preflight else 'screen'}", output_dir=str(Path(output) / "runs"), wandb_project=None)
    cfg["experiment"] = {"name": "MAIN_WEIGHT_EARLY_SCREEN_V1", "arm": arm, "preflight_discarded": preflight}
    cfg["training"].update(max_steps=steps, schedule_steps=10000, batch_size=64, gradient_accumulation=1,
        warmup_ratio=.05, lr=2e-4, grad_clip=1., log_every=50, eval_every=500 if preflight else 250,
        save_steps=[2] if preflight else [250, 500, 1000], selection_steps=[] if preflight else [250, 500, 1000], validation_batches=0,
        validation_snrs=["no_noise"], patience=None, max_minutes=None,
        loss_weights={"kl": 1., "hidden": ARMS[arm], "memory": .05},
        streamed_backward=True, valid_only_kl=True,
        presentation_stream={"policy": "epoch-permutations-v1", "seed": 0, "total_presentations": steps * 64},
        paired_randomness={"policy": "step-microbatch-stream-v1", "seed": 0, "audit_steps": list(range(1, steps+1))})
    cfg["training"].pop("feature_summary", None)
    validate_config(cfg)
    return cfg


def frozen_selection_order(saved_ids, fixture):
    """Validate E3 membership, then retain its canonical second-shuffle order."""
    rows = fixture["rows"]
    canonical = validate_fixture(rows, saved_ids)
    ids = [row["row_id"] for row in canonical]
    if ids != [row["row_id"] for row in rows]:
        raise ValueError("frozen fixture does not use canonical E3 presentation order")
    return ids


def run_screen(base, output, arm, *, preflight=False, seconds=2600, deadline_epoch=None, selection_fixture=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = screen_config(base, output, arm, preflight)
    cap = cfg["training"]["max_steps"]
    started = time.monotonic()
    report = {"arm": arm, "status": "RUNNING", "preflight_discarded": preflight,
              "optimizer_calls": 0, "training_presentations": 0, "selection_presentations": 0,
              "weight_policy": "explicit loss_weights overrides legacy kl_weight/mse_weight",
              "loss_weights": cfg["training"]["loss_weights"], "execution_flags": {k: cfg["training"][k] for k in ("streamed_backward", "valid_only_kl")},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    original_build, original_load, original_loss = training.build_model, training.load_data, training.batch_losses
    original_step, original_scheduler, original_validate = torch.optim.AdamW.step, training.make_scheduler, training.validate
    active = {}

    def flush():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def deadline():
        if time.monotonic() - started >= seconds or (deadline_epoch is not None and time.time() >= deadline_epoch):
            raise TimeoutError("bounded screen deadline reached; preserve partial checkpoint, do not retry")

    def build(config):
        processor, model = original_build(config)
        if model.memory_codec is None:
            raise ValueError("receiver-memory codec is mandatory")
        active["model"] = model
        report["initial_state"] = {name: {k: tensor_digest(v) for k, v in module.state_dict().items()}
                                   for name, module in (("hidden", model.codec), ("memory", model.memory_codec))}
        return processor, model

    def load(*args, **kwargs):
        data = original_load(*args, **kwargs)
        ids = selection_ids(data.ids)
        if selection_fixture is not None:
            fixture = json.loads(Path(selection_fixture).read_text())
            ids = frozen_selection_order(data.ids, fixture)
            report["selection_fixture_sha256"] = hashlib.sha256(Path(selection_fixture).read_bytes()).hexdigest()
        positions = [data.ids["validation_rows"].index(row) for row in ids]
        loader = data.validation
        data.validation = DataLoader(Subset(loader.dataset, positions), batch_size=64, shuffle=False,
                                      collate_fn=loader.collate_fn, num_workers=0)
        report["selection_ids"] = ids
        active["data"] = data
        return data

    def loss(model, batch, settings, snr, **kwargs):
        deadline()
        phase = "training_presentations" if "row_ids" in batch else "selection_presentations"
        count = batch["labels"].shape[0]
        if report[phase] + count > (cap*64 if phase.startswith("training") else 512):
            raise RuntimeError("screen presentation budget exhausted")
        report[phase] += count
        native, _ = training.model_inputs(batch, model)
        record = {"phase": phase, "step": report["optimizer_calls"],
                  "batch": {k: v.cpu().tolist() for k, v in batch.items() if torch.is_tensor(v)},
                  "native_decoder_input_ids": native["decoder_input_ids"].cpu().tolist(),
                  "native_decoder_sha256": tensor_digest(native["decoder_input_ids"]),
                  "snr": snr.detach().cpu().reshape(-1).tolist() if torch.is_tensor(snr) else snr,
                  "snr_sha256": tensor_digest(snr) if torch.is_tensor(snr) else None}
        with gzip.open(output / "presented_native_inputs.jsonl.gz", "at") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return original_loss(model, batch, settings, snr, **kwargs)

    def validate(model, loader, config):
        if preflight:
            return {"loss": 0., "kl": 0., "nmse": 0.}  # no selection GPU work in discarded preflight
        result = original_validate(model, loader, config)
        with (output / "selection_components.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": report["optimizer_calls"], **result}, allow_nan=False) + "\n")
        return result

    def scheduler(optimizer, settings):
        result = original_scheduler(optimizer, settings)
        model, data = active["model"], active["data"]
        report["initial_state_fp32"] = {name: {k: tensor_digest(v) for k, v in module.state_dict().items()}
                                        for name, module in (("hidden", model.codec), ("memory", model.memory_codec))}
        torch.save({"codec": model.codec.state_dict(), "memory_codec": model.memory_codec.state_dict(),
                    "channel": model.channel.state_dict(), "config": cfg, "data_ids": data.ids, "step": 0}, output / "init.pt")
        if preflight:
            # Two discarded nonzero-LR updates; formal schedule remains unchanged.
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"]
        else:
            validate(model, data.validation, cfg)
        return result

    def step(optimizer, *args, **kwargs):
        deadline()
        if report["optimizer_calls"] >= cap:
            raise RuntimeError("optimizer update cap exceeded")
        model = active["model"]
        if preflight:
            for group in optimizer.param_groups:
                group["lr"] = cfg["training"]["lr"]
        norms = {}
        for name, module in (("hidden", model.codec), ("memory", model.memory_codec)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError(f"missing/nonfinite {name} gradients")
            norms[name] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
            if norms[name] == 0:
                raise FloatingPointError(f"zero {name} gradient")
        result = original_step(optimizer, *args, **kwargs)
        report["optimizer_calls"] += 1
        if preflight:
            states = [value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
            if not states or not all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in states):
                raise FloatingPointError("Adam states must be finite FP32")
            with (output / "preflight_updates.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": report["optimizer_calls"], "lr": optimizer.param_groups[0]["lr"], "postclip_gradient_norms": norms, "adam_states_fp32_finite": True}) + "\n")
        flush()
        return result

    try:
        with ExitStack() as stack:
            for target, name, replacement in ((training, "build_model", build), (training, "load_data", load),
                (training, "batch_losses", loss), (training, "validate", validate),
                (training, "make_scheduler", scheduler), (torch.optim.AdamW, "step", step)):
                stack.enter_context(patch.object(target, name, replacement))
            report["run"] = str(training.train(cfg))
        if report["optimizer_calls"] != cap or report["training_presentations"] != cap * 64 or report["selection_presentations"] != (0 if preflight else 512):
            raise RuntimeError("screen counts differ from fixed budget")
        report["status"] = "MEASURED"
    except BaseException as exc:
        report.update(status="FAILED_OR_PARTIAL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        flush()
    return report


def prepare_reconstruction_fixture(base_path: str, output_dir: str | Path, *, total_updates: int,
                 local_updates: int, functional_updates: int,
                 functional_lr: float | None = None, include_reset_control: bool = False) -> dict:
    """Export a coherent four-arm plan; all budget numbers are caller supplied."""
    if min(total_updates, local_updates, functional_updates) < 1 or total_updates != local_updates + functional_updates:
        raise ValueError("require positive explicit U, L and F with U = L + F")
    base = load_config(base_path)
    from jscc.checkpoint_readers import parent_recipe_sha256
    check_reconstruction_recipe(base)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("plan directory must be fresh and empty")
    output.mkdir(parents=True, exist_ok=True)
    batch = base["training"]["batch_size"] * base["training"]["gradient_accumulation"]
    arm_paths = {}
    b_digest = None
    for arm, updates, start in (("A", total_updates, 0), ("B", local_updates, 0),
                                ("C", functional_updates, local_updates * batch),
                                ("D", functional_updates, local_updates * batch)):
        config = copy.deepcopy(base)
        settings = config["training"]
        settings["max_steps"] = settings["schedule_steps"] = updates
        settings["eval_every"] = updates
        settings["min_steps"] = updates
        settings["patience"] = None
        settings["save_steps"] = [updates]
        settings["selection_steps"] = [updates]
        settings["feature_summary"] = None
        settings["presentation_stream"]["total_presentations"] = updates * batch
        settings["presentation_stream"]["start_presentation"] = start
        settings["paired_randomness"]["audit_steps"] = [1, updates]
        if arm in ("C", "D") and functional_lr is not None:
            if not math.isfinite(functional_lr) or functional_lr <= 0:
                raise ValueError("functional LR must be finite and positive")
            settings["lr"] = functional_lr
        if arm == "B":
            b_digest = parent_recipe_sha256(config)
        if arm == "C":
            assert b_digest is not None
            settings["phase_transfer"] = {"parent_recipe_sha256": b_digest,
                                          "parent_step": local_updates}
        config["run"]["name"] = f"hellaswag-enc_l9-{arm.lower()}"
        config["run"]["output_dir"] = str(output / "runs")
        path = output / f"{arm.lower()}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arm_paths[arm] = str(path)
    if include_reset_control:
        e1 = load_config(arm_paths["B"])
        e1["run"]["name"] = "hellaswag-enc_l9-e1"
        e = load_config(arm_paths["C"])
        e["run"]["name"] = "hellaswag-enc_l9-e"
        e["training"]["phase_transfer"] = {
            "parent_kind": "functional", "parent_recipe_sha256": parent_recipe_sha256(e1),
            "parent_step": local_updates}
        for arm, config in (("E1", e1), ("E", e)):
            path = output / f"{arm.lower()}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            arm_paths[arm] = str(path)
    result = check_reconstruction_plan(arm_paths)
    result["recipes"] = arm_paths
    result["base_recipe"] = str(Path(base_path).resolve())
    result["execution"] = "none; use explicit capture/local/functional commands with --max-seconds"
    (output / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def prepare_coco_native_fixture(measurements, source_root, evidence_root, output, study_spec=None):
    receipt = json.loads(Path(measurements).read_text())
    updates = coco_native_validate_measurements(receipt)
    coco_native_verify_bindings(receipt, evidence_root, source_root)
    if study_spec is None:
        study_spec = Path(__file__).resolve().parents[1] / "configs/studies/splits.yaml"
    base = expand_study(study_spec, task="coco")
    if tuple(run.experiment for run in base.runs) != coco_native_SPLITS:
        raise ValueError("The study must contain the canonical thirteen REF routes in order")
    for run in base.runs:
        config = run.config
        if run.seed != 0 or config["codec"]["snr_film"] or not config["channel"]["normalize_power"]:
            raise ValueError("Native baseline requires seed0, FiLM off and retained power normalization")
        expected_norm = "post" if run.experiment == "enc_emb" else "both" if run.experiment == "enc_fn" else "none"
        if config["codec"]["layernorm"] != expected_norm or config["codec"].get("memory", {}).get("layernorm") != "both":
            raise ValueError("REF main/memory outer LayerNorm policy changed")
        training = config["training"]
        if coco_native_integer(config["data"]["num_train"], "training row count") % receipt["training_batch_size"]:
            raise ValueError("Microbatch must divide data.num_train; larger batches require an explicit full-batch sampling policy before approval")
        # A graceful timer stop is partial evidence, not successful fixed-budget training.
        training.pop("max_minutes", None)
        training.update(batch_size=receipt["training_batch_size"],
                        gradient_accumulation=receipt["gradient_accumulation"],
                        max_steps=updates, schedule_steps=updates, eval_every=updates,
                        selection_steps=[updates], save_steps=[updates], patience=None,
                        streamed_backward=True, valid_only_kl=True)
        config["evaluation"].update(snrs=coco_native_CONDITIONS, vanilla=False, mode="codec_only",
                                    batch_size=receipt["evaluation_batch_size"])
        config["run"]["name"] = f"coco-native-ref13-{run.experiment}-s0"
        validate_config(config)
    samples = base.runs[0].config["evaluation"]["num_samples"]
    timing = coco_native_estimate(receipt, updates, samples)
    manifest = export_study(StudyPlan("coco-native-ref13", base.runs), output)
    output = manifest.parent
    data = json.loads(manifest.read_text())
    for row in data["runs"]:
        row["evaluation_link"] = f"links/{row['index']:04d}-evaluation.txt"
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    vanilla = copy.deepcopy(base.runs[0].config["evaluation"])
    vanilla.update(mode="vanilla_only", vanilla=True)
    save_config(vanilla, output / "vanilla.yaml")
    (output / "preflight-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    card = {"version": 1, "execution_authorized": False,
            "numerical_limitations": coco_native_numerical_limitations(receipt), "accept_numerical_limitations": False,
            "status": "AWAITING_TIMING_REVIEW_AND_RENEWED_APPROVAL", "source_root": str(Path(source_root).resolve()),
            "approval_reference": None, "site_four_gpu_cap_verified": False,
            "selected_indices": coco_native_selected_indices(receipt),
            "runs": len(coco_native_selected_indices(receipt)), "panels": 4 * len(coco_native_selected_indices(receipt)) + 1,
            "presentations_per_run": receipt.get("presentations_per_run", coco_native_PRESENTATIONS),
            "updates": updates, "horizon": updates, "conditions": coco_native_CONDITIONS,
            "training_batch_size": receipt["training_batch_size"],
            "gradient_accumulation": receipt["gradient_accumulation"],
            "evaluation_batch_size": receipt["evaluation_batch_size"], "images_per_panel": samples,
            "concurrency": "one GPU/job; site account cap four GPUs must be verified before future dispatch",
            "fixed_checkpoint": f"step_{updates:06d}.pt", "estimates": timing,
            "decision_scenarios": coco_native_scenarios(receipt),
            "selection_policy": {"candidate": "final_only", "locked": False,
                "historical_policy": "periodic_validation",
                "note": "Prospective timing scenario; selection cadence and its cost require a decision before formal configuration freeze."},
            "training_execution": {"streamed_backward": True, "valid_only_kl": True},
            "graph": f"{len(coco_native_selected_indices(receipt))} selected train array tasks; same-index aftercorr evaluations; one shared vanilla after enc_fn train",
            "submission": "Dry-run commands only; execution requires frozen renewed approval and verified site concurrency cap."}
    (output / "APPROVAL-CARD.json").write_text(json.dumps(card, indent=2) + "\n")
    bindings = {str(path.relative_to(output)): coco_native_digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "PREPARED-FILES.json").write_text(json.dumps(bindings, indent=2) + "\n")
    return card


native_policy_PRESENTATION_CAP = 32768


native_policy_UPDATE_CAP = 512


def native_policy_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_policy_matrix():
    return [(f"{repeat}-b{batch}", batch, 5120, repeat, False)
            for repeat, batches in (("A", (64, 128)), ("B", (128, 64))) for batch in batches] + [
        (f"stress-b{batch}", batch, batch * 2, "stress", True) for batch in (64, 128)] + [("profile-b64", 64, 128, "profile", True)]


class native_policy_BudgetDeadline(TimeoutError):
    """Only the authorized benchmark deadline, not a model/data I/O timeout."""


class native_policy_Budget:
    """Charge before dispatch, including unsuccessful attempted forwards/updates."""
    def __init__(self, seconds=3300):
        self.presentations = 0
        self.updates = 0
        self.deadline = time.monotonic() + seconds

    def charge(self, examples=0, updates=0):
        if time.monotonic() >= self.deadline:
            raise native_policy_BudgetDeadline("route time limit reached; no retry")
        if self.presentations + examples > native_policy_PRESENTATION_CAP or self.updates + updates > native_policy_UPDATE_CAP:
            raise RuntimeError("route work cap exceeded before dispatch")
        self.presentations += examples
        self.updates += updates


def native_policy_order_for(dataset, repeat, count):
    indices = list(range(len(dataset)))
    if repeat == "stress":
        indices.sort(key=lambda i: (len(dataset[i]["input_ids"]) + len(dataset[i]["label_ids"]), i), reverse=True)
    else:
        random.Random(0 if repeat == "A" else 1).shuffle(indices)
    if len(indices) < count:
        raise ValueError("insufficient distinct rows for the declared paired workload")
    return indices[:count]


def native_policy_candidate_config(base, output, name, batch, count):
    config = copy.deepcopy(base)
    config["run"].update(name=name, output_dir=str(output / "runs"), wandb_project=None)
    config["training"].update(max_steps=count // batch, batch_size=batch, gradient_accumulation=1,
        schedule_steps=20000, warmup_ratio=0.05, lr=2e-4, eval_every=count // batch,
        save_steps=[count // batch], validation_batches=0, streamed_backward=True,
        valid_only_kl=True, low_sync_logging=False, deterministic_algorithms=True,
        patience=None, max_minutes=None)
    return config


def native_policy_validate_policy(config):
    if config.get("protocol") != "corrected-baseline-v2-native-decoder-inputs":
        raise ValueError("requires native-v2 protocol")
    hidden, memory = resolve_codec_configs(config["codec"])
    for codec in (hidden, memory):
        if codec["snr_film"] or (codec["hidden_dim"], codec["bottleneck_dim"], codec["n_res_blocks"]) != (1152, 512, 2):
            raise ValueError("requires FiLM off and H1152/B512/two blocks on both streams")
    if hidden["layernorm"] != "none" or memory["layernorm"] != "both":
        raise ValueError("requires raw-layer main none / memory both normalization")
    if config["task"] != "hellaswag" or config["data"]["num_validation"] != 512:
        raise ValueError("requires the fixed HellaSwag 512-row selection")
    if config["data"]["max_length"] != 512:
        raise ValueError("long-input stress requires the unchanged max_length512")
    channel = config["channel"]
    if channel["type"] != "awgn" or not channel["train_noise"] or channel["train_snr_range"] != [-6, 18]:
        raise ValueError("requires production Uniform[-6,18] AWGN")
    if config["training"]["validation_snrs"] != ["no_noise"]:
        raise ValueError("exactly one no_noise selection pass is allowed")
    split = config["split"]
    if (split["stack"], split["index"]) not in (("enc", 9), ("dec", 8)) or split["where"] != "after_layer":
        raise ValueError("only encoder layer9 or decoder layer8 is authorized")
    if config["model"]["device"] != "cuda" or config["model"]["dtype"] != "bfloat16":
        raise ValueError("requires CUDA with BF16 frozen backbone")


def native_policy_start_profile(report):
    """Trace tooling is optional; its setup failure is not a training failure."""
    try:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True)
        profiler.start()
    except Exception as exc:
        report["profiler"] = {"status": "unavailable", "stage": "start", "error": f"{type(exc).__name__}: {exc}"}
        return None
    report["profiler"] = {"status": "running"}
    return profiler


def native_policy_finish_profile(profiler, output, report):
    """Catch only tool stop/export errors, outside actual training operations."""
    if profiler is None:
        return
    try:
        profiler.stop()
        profiler.export_chrome_trace(str(output / "trace.json"))
    except Exception as exc:
        report["profiler"] = {"status": "partial_or_unavailable", "stage": "stop_or_export",
                              "error": f"{type(exc).__name__}: {exc}",
                              "partial_trace_exists": (output / "trace.json").exists()}
        return
    report["trace"] = str(output / "trace.json")
    report["profiler"] = {"status": "complete"}



def native_policy_run_candidate(config, output, descriptor, budget, select, expected_initial=None):
    name, batch_size, count, repeat, stress = descriptor
    report = {"name": name, "status": "RUNNING", "selection": "pending" if select else "suppressed",
              "config_sha256": native_policy_digest(config), "attempted_presentations": 0, "optimizer_calls": 0,
              "completed_updates": 0, "selection_presentations": 0, "phase_seconds": {},
              "evidence_write_seconds_nested": 0.,
              "noise_policy": "production SNR per sample; same SNR shared across streams, fresh channel draws per stream"}
    output.mkdir()
    save_config(config, output / "config.yaml")
    records = output / "batches.jsonl"
    started = time.perf_counter()
    phase = "setup"
    phase_started = started
    held = {}
    consumed = []
    measured_lengths = {"input": [], "target": []}
    profiling = repeat == "profile"
    original_load, original_prepare = training.load_data, training.prepare_trainable_parameters
    original_loss, original_validate, original_save = training.batch_losses, training.validate, training.save_checkpoint
    original_step, original_clip = torch.optim.AdamW.step, torch.nn.utils.clip_grad_norm_

    def flush():
        write_started = time.perf_counter()
        report["route_charged"] = {"presentations": budget.presentations, "updates": budget.updates}
        temp = output / "report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "report.json")
        report["evidence_write_seconds_nested"] += time.perf_counter() - write_started

    def append_record(record):
        write_started = time.perf_counter()
        with records.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        report["evidence_write_seconds_nested"] += time.perf_counter() - write_started

    def transition(next_phase):
        nonlocal phase, phase_started
        torch.cuda.synchronize()
        now = time.perf_counter()
        report["phase_seconds"][phase] = report["phase_seconds"].get(phase, 0) + now - phase_started
        phase, phase_started = next_phase, now

    def prepare(model):
        original_prepare(model)
        held["model"] = model
        state = hashlib.sha256()
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                state.update(key.encode())
                state.update(parameter.detach().cpu().float().numpy().tobytes())
        held["initial"] = {key: parameter.detach().cpu().clone() for key, parameter in model.named_parameters()
                           if parameter.requires_grad}
        report["initial_parameters_sha256"] = state.hexdigest()
        if expected_initial is not None and state.hexdigest() != expected_initial:
            raise RuntimeError("fresh candidate initialization differs")
        if profiling:
            original_forward = model.forward
            def forward(*args, **kwargs):
                with torch.profiler.record_function("teacher" if model.bypass else "student"):
                    return original_forward(*args, **kwargs)
            model.forward = forward
        report["backend"] = str(getattr(model.base.config, "_attn_implementation", None))

    def load(*args, **kwargs):
        data = original_load(*args, **kwargs)
        order = native_policy_order_for(data.train.dataset, repeat, count)
        report["planned_ids"] = [data.ids["train_rows"][i] for i in order]
        report["planned_ids_sha256"] = native_policy_digest(report["planned_ids"])
        report["selection_ids_sha256"] = native_policy_digest(data.ids["validation_rows"])
        data.train = DataLoader(IndexedRows(data.train.dataset, data.ids["train_rows"]),
            batch_size=batch_size, sampler=order, num_workers=config["data"]["num_workers"],
            collate_fn=IndexedCollator(data.train.collate_fn), pin_memory=True)
        flush()
        return data

    def losses(model, batch, settings, snr, **kwargs):
        ids = batch.get("preflight_row_ids")
        if ids is None:
            budget.charge()
            report["selection_presentations"] += len(batch["labels"])
            if report["selection_presentations"] > 512:
                raise RuntimeError("selection cap exceeded")
            return original_loss(model, batch, settings, snr, **kwargs)
        if phase == "setup":
            transition("stress" if stress else "warmup")
            torch.cuda.reset_peak_memory_stats()
            if profiling:
                held["profiler"] = native_policy_start_profile(report)
        elif not stress and report["attempted_presentations"] == 1024:
            transition("measured")
        budget.charge(examples=len(ids))
        report["attempted_presentations"] += len(ids)
        consumed.extend(ids)
        record = {"row_ids": ids, "phase": phase, "input_lengths": batch["attention_mask"].sum(1).tolist(),
                  "target_lengths": (batch["labels"] != -100).sum(1).tolist(),
                  "input_shape": list(batch["input_ids"].shape), "target_shape": list(batch["labels"].shape),
                  "snr_db": snr.detach().cpu().tolist(), "status": "attempted"}
        if phase == "measured":
            measured_lengths["input"].extend(record["input_lengths"])
            measured_lengths["target"].extend(record["target_lengths"])
        record["stream_snr_db"] = {"hidden": record["snr_db"]}
        if model.memory_codec is not None:
            record["stream_snr_db"]["memory"] = record["snr_db"]
        record["padding_tokens"] = {
            "input": batch["input_ids"].numel() - sum(record["input_lengths"]),
            "target": batch["labels"].numel() - sum(record["target_lengths"])}
        # Persist attempted rows before a forward can fail or an allocation ends.
        append_record(record)
        flush()
        with torch.profiler.record_function("batch_loss") if profiling else nullcontext():
            result = original_loss(model, batch, settings, snr, **kwargs)
        record = {"batch": report["attempted_presentations"] // batch_size, "status": "forward_complete",
                  "payload_allocated": {k: int(v) for k, v in model.channel_uses_allocated.items()},
                  "payload_valid": {k: int(v) for k, v in model.valid_payload_counts.items()}}
        append_record(record)
        if not all(torch.isfinite(result[k]).all().item() for k in ("loss", "kl", "nmse")):
            raise FloatingPointError("nonfinite production loss")
        return result

    def clip(*args, **kwargs):
        report["gradient_names"] = checked_gradient_names(held["model"])
        kwargs["error_if_nonfinite"] = True
        return original_clip(*args, **kwargs)

    def step(optimizer, *args, **kwargs):
        if report["optimizer_calls"] == 0 and optimizer.state:
            raise RuntimeError("optimizer must start empty")
        budget.charge(updates=1)
        report["optimizer_calls"] += 1
        flush()
        result = original_step(optimizer, *args, **kwargs)
        report["completed_updates"] += 1
        report["precision"] = training.precision_telemetry(held["model"], optimizer)
        return result

    def validate(*args, **kwargs):
        native_policy_finish_profile(held.pop("profiler", None), output, report)
        budget.charge()
        transition("selection" if select else "finalize")
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if not select:
            # Production loop requires a metric dict to finish. These sentinels
            # are explicitly excluded from quality reports and no best is saved.
            return {"loss": 0., "kl": 0., "nmse": 0.}
        result = original_validate(*args, **kwargs)
        if report["selection_presentations"] != 512:
            raise RuntimeError("selection did not consume exactly512")
        if not all(torch.isfinite(torch.tensor(v)).item() for v in result.values()):
            raise FloatingPointError("nonfinite selection metric")
        report["selection"] = "complete"
        report["selection_metrics"] = result
        transition("finalize")
        return result

    def save(path, *args, **kwargs):
        if not Path(path).name.startswith("step_"):
            return None
        budget.charge()
        transition("checkpoint")
        result = original_save(path, *args, **kwargs)
        report["checkpoint"] = str(path)
        transition("finalize")
        return result

    original_backward = torch.autograd.backward
    def backward(*args, **kwargs):
        with torch.profiler.record_function("backward"):
            return original_backward(*args, **kwargs)

    original_kl = training.distillation_loss_stats
    def kl(*args, **kwargs):
        with torch.profiler.record_function("loss_kl"):
            return original_kl(*args, **kwargs)

    try:
        budget.charge()  # Reject exhausted time before model/data setup.
        with ExitStack() as stack:
            if profiling:
                stack.enter_context(patch.object(torch.autograd, "backward", backward))
                stack.enter_context(patch.object(training, "distillation_loss_stats", kl))
            for target, attr, replacement in ((training, "load_data", load),
                (training, "prepare_trainable_parameters", prepare), (training, "batch_losses", losses),
                (training, "validate", validate), (training, "save_checkpoint", save),
                (torch.optim.AdamW, "step", step), (torch.nn.utils, "clip_grad_norm_", clip)):
                stack.enter_context(patch.object(target, attr, replacement))
            report["run"] = str(training.train(config))
        if report["completed_updates"] != count // batch_size:
            raise RuntimeError("production loop stopped before declared work completed")
        report["update_l2"] = sum(
            (parameter.detach().cpu().float() - held["initial"][key].float()).square().sum().item()
            for key, parameter in held["model"].named_parameters() if parameter.requires_grad) ** 0.5
        if not 0 < report["update_l2"] < float("inf"):
            raise FloatingPointError("no finite nonzero parameter update")
        precision = report["precision"]
        for group, dtype in (("trainable_parameters", "float32"), ("frozen_parameters", "bfloat16"),
                             ("optimizer_state", "float32")):
            if set(precision[group]) != {dtype}:
                raise RuntimeError(f"unexpected {group} precision")
        report["status"] = "PASS"
    except native_policy_BudgetDeadline as exc:
        report.update(status="TIME_LIMIT", error=f"{type(exc).__name__}: {exc}")
    except torch.OutOfMemoryError as exc:
        report.update(status="OOM", error=str(exc))
    except BaseException as exc:
        report.update(status="STOP_ROUTE", error=f"{type(exc).__name__}: {exc}")
    finally:
        native_policy_finish_profile(held.pop("profiler", None), output, report)
        try:
            transition("done")
        except RuntimeError:
            report["timing_limitation"] = "CUDA synchronization failed after error"
        report["consumed_ids_sha256"] = native_policy_digest(consumed)
        if report["status"] == "PASS" and not stress:
            seconds = report["phase_seconds"]["measured"]
            report["measured_examples_per_second"] = 4096 / seconds
            report["measured_valid_tokens_per_second"] = sum(sum(v) for v in measured_lengths.values()) / seconds
            report["measured_length_quantiles"] = {
                key: torch.tensor(values, dtype=torch.float32).quantile(torch.tensor([0., .5, .9, .99, 1.])).tolist()
                for key, values in measured_lengths.items()}

        report["total_seconds"] = time.perf_counter() - started
        flush()
        held.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return report
