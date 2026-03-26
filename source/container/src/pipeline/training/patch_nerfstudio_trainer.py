#!/usr/bin/env python3
"""
Patch for nerfstudio trainer.py:
Fix optimizer.state_dict() KeyError when save_checkpoint is called after
Gaussian densification replaces parameter tensors (new id()s not in param_mappings).
"""
import sys

OLD = '            "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},'
NEW = '''            "optimizers": {k: _safe_optimizer_state_dict(v) for (k, v) in self.optimizers.optimizers.items()},'''

HELPER = '''
def _safe_optimizer_state_dict(optimizer):
    """Return optimizer state_dict, or empty dict if param_mappings are stale.
    This happens with Gaussian Splatting when densification replaces parameter
    tensors in-place, leaving the optimizer holding old tensor ids."""
    try:
        return optimizer.state_dict()
    except KeyError:
        print("[WARNING] optimizer.state_dict() failed (stale param ids after densification) - saving empty optimizer state")
        return {}

'''

def patch_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        if OLD not in content:
            print(f"⚠ Target line not found in {filepath}")
            return False

        if '_safe_optimizer_state_dict' in content:
            print(f"⚠ Already patched: {filepath}")
            return True

        # Insert helper before the Trainer class definition
        content = content.replace(
            '\nclass Trainer:',
            HELPER + '\nclass Trainer:'
        )
        content = content.replace(OLD, NEW)

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"✓ Successfully patched {filepath}")
        return True

    except Exception as e:
        print(f"✗ Error patching {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_nerfstudio_trainer.py <trainer.py path>")
        sys.exit(1)
    success = patch_file(sys.argv[1])
    sys.exit(0 if success else 1)
