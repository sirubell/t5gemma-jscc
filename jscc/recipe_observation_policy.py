"""Resolve save states and fixed observation clocks for complete location cycles."""
import math


def checkpoint_policy(quota, ratio, n, policy):
    recon = quota * ratio // 100
    functional = quota - recon
    if set(policy) != {'reconstruction_fractions', 'functional_fractions', 'functional_observation_fraction'}:
        raise ValueError('save policy requires reconstruction/functional fractions and observation fraction')
    result = [(0, 'initialization')]
    for phase, q, start in [('reconstruction', recon, 0), ('functional', functional, recon)]:
        fractions = policy[f'{phase}_fractions']
        if (not isinstance(fractions, list) or not fractions or fractions[-1] != 1
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in fractions)
                or fractions != sorted(set(fractions))):
            raise ValueError('save fractions must be increasing, unique and end at 1')
        for fraction in fractions:
            offset = q * fraction
            if not float(offset).is_integer():
                raise ValueError('fractional per-location save quota is unsupported')
            role = ('pre_reset' if phase == 'reconstruction' else 'terminal') if fraction == 1 else phase
            result.append(((start + int(offset)) * n, role))
    fraction = policy['functional_observation_fraction']
    observation = []
    if fraction is not None:
        if type(fraction) not in (int, float) or not math.isfinite(fraction):
            raise ValueError('functional observation fraction must be finite')
        if not 0 < fraction < 1 or not float(functional * fraction).is_integer():
            raise ValueError('fractional per-location observation quota is unsupported')
        step = (recon + int(functional * fraction)) * n
        result = [row for row in result if row[0] != step]
        result.append((step, 'functional_observation'))
        observation.append((step, 'functional_observation'))
    result.append((recon * n, 'post_reset'))
    # Pre and post reset deliberately remain distinct at the same update.
    result.sort(key=lambda value: (value[0], value[1] == 'post_reset'))
    return [list(row) for row in result], [(recon * n, 'pre_reset'), *observation]



def scoring_template(template, policy, evaluation):
    """Bind fixed scorer inputs to resolved YAML settings before executing panels."""
    from copy import deepcopy
    fields = ('task', 'input_ids', 'source_family_ids', 'settings', 'noise',
              'data_settings', 'scorer', 'precision', 'backend', 'reference_corpus', 'prompt_policy', 'layout')
    value = {k: deepcopy(template[k]) for k in fields}
    count = policy['items']
    settings = value['settings']
    settings.update(num_samples=count, batch_size=evaluation['candidate_batch'],
                    num_fewshot=evaluation['fewshot'], scoring_policy=evaluation['scoring_policy'])
    value['expected_items'] = count
    if (value['task'] != 'hellaswag' or value['input_ids'] != list(range(count))
            or len(value['source_family_ids']) != count
            or any(not isinstance(f, str) or not f for f in value['source_family_ids'])
            or settings.get('numerical_policy', 'native') != 'native'
            or settings.get('vanilla') is not False
            or value['scorer'] != 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm'
            or value['reference_corpus']['split'] != 'validation'
            or type(value['noise']['seed']) is not int or not value['noise']['namespace']):
        raise ValueError('recipe fixed scoring inputs/order/native policy differ')
    return value


def recipe_observation_identity(description, template):
    """Checkpoint compatibility binds fixed scorer inputs as well as YAML policy."""
    from .activation_replay import canonical_digest
    return canonical_digest([{'policy': policy, 'template': scoring_template(
        template, policy, description['evaluation'])} for policy in description['observations']])
