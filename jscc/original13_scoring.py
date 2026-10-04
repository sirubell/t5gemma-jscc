"""Original13 task scoring and isolated runtime restoration."""
from contextlib import contextmanager
import copy

import torch

from .experiment_state import capture_rng, restore_rng
from .sharing_qualification_guard import state_exact
from .sharing_preparation import state_dict_identity


@contextmanager
def _original13_runtime(model, state):
    """Restore canonical weights, gradients, global RNG and all idle route state.

    The routing owner provides the model-side idle-state contract. Validation and
    restoration errors are fatal, including when a scorer/output error also occurs.
    """
    from .encoder_stages_evaluation import _reopen_original13
    from . import original13_routing as routing
    codec = routing.canonical_codec(model)
    previous_weights = copy.deepcopy(routing.export_codec(model))
    previous_route = copy.deepcopy(routing.observation_state(model))
    parameters = list(codec.parameters())
    grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    rng = capture_rng()
    try:
        routing.load_codec(model, state.payload['model'])
        model.eval()
        yield
    finally:
        try:
            unchanged = state_dict_identity(routing.export_codec(model)) == state_dict_identity(state.payload['model'])
        finally:
            # Even integrity inspection itself can fail. Always attempt route and
            # RNG cleanup; restoration failure remains fatal and may supersede it.
            try:
                routing.load_codec(model, previous_weights)
            finally:
                try:
                    for parameter, grad in zip(parameters, grads, strict=True):
                        parameter.grad = grad
                    routing.restore_runtime_state(model, previous_route)
                finally:
                    restore_rng(rng)
        if not unchanged:
            raise ValueError('original13 scorer mutated checkpoint weights')
        reopened, _ = _reopen_original13(state)
        if (not state_exact(reopened.payload, state.payload)
                or not state_exact(routing.export_codec(model), previous_weights)
                or not state_exact(routing.observation_state(model), previous_route)
                or not state_exact(capture_rng(), rng)):
            raise ValueError('original13 observation restoration/checkpoint integrity differs')


@contextmanager
def _original13_scoring_policy(settings):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(settings.get('deterministic_algorithms', enabled))
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _score_original13_panel(model, processor, *, state, request, site_id, condition, output):
    """Use the real task adapter, with no codec bypass in the no_noise condition."""
    from .encoder_stages_evaluation import ObservationScoringError
    from .evaluation import evaluate_hellaswag, _observation_items, TaskScoringError
    from .models import split_model
    from .experiment_state import isolated_rng
    site = getattr(split_model, "resolve_original13_site")(model.base, site_id, state.payload['metadata']['model_revision'])
    settings = {**request['settings'], 'noise_seed': request['noise']['seed'],
                'noise_namespace': request['noise']['namespace'], 'evidence_mode': 'compact-v1'}
    dtype = next(model.base.parameters()).dtype
    dtype_names = {torch.float32: 'fp32', torch.bfloat16: 'bf16', torch.float16: 'fp16'}
    backend = getattr(model.base.config.decoder, '_attn_implementation', None)
    if (request['precision'] not in (str(dtype), dtype_names.get(dtype))
            or request['backend'] != backend
            or getattr(model, 'numerical_policy', 'native') != 'native'):
        raise ValueError('original13 actual model precision/backend differs')
    snr = None if condition == 'no_noise' else float(condition)
    with _original13_scoring_policy(settings), isolated_rng(settings['noise_seed']), model.channel.replay(
            settings['noise_seed'], settings['noise_namespace']), model.at_site(
            site, purpose='evaluate', authorization={'sites': {site_id: 'trained'}}), model.transmission(snr):
        try:
            metrics = evaluate_hellaswag(model, processor, settings, output, condition, request['data_settings'])
        except TaskScoringError as error:
            raise ObservationScoringError(str(error)) from error
        payload = {'allocated': dict(model.channel_uses_allocated), 'valid': dict(model.valid_payload_counts)}
    items, ids, families = _observation_items(output, 'hellaswag', condition)
    if ids != request['input_ids'] or families != request['source_family_ids']:
        raise ValueError('original13 observed question/source-family order differs')
    if not metrics.get('finite_scores') or metrics['candidate_forward_requests'] != 4 * request['expected_items']:
        raise ValueError('original13 finite score/candidate denominator differs')
    return {'completed_items': len(items), 'denominator': len(items), 'metrics': metrics, 'payload': payload}

