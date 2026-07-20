"""
Patch torch.optim.Optimizer.state_dict to purge stale state entries.

After gsplat densification (split/prune), removed parameter tensors leave
dangling entries in optimizer.state keyed by the tensor object. PyTorch's
state_dict() builds param_mappings only from current param_groups, so stale
keys cause a KeyError. This patch removes them before serialization.
"""

import torch


_original_state_dict = torch.optim.Optimizer.state_dict


def _safe_state_dict(self):
    valid_params = {id(p) for group in self.param_groups for p in group["params"]}
    stale = [k for k in self.state if isinstance(k, torch.Tensor) and id(k) not in valid_params]
    for k in stale:
        del self.state[k]
    return _original_state_dict(self)


torch.optim.Optimizer.state_dict = _safe_state_dict
