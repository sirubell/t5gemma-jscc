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

