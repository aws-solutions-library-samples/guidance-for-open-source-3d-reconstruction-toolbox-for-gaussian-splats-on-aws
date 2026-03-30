# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""
Patch tracer.py to add cache path for compiled extensions
"""

import sys
for fpath in sys.argv[1:]:
 with open(fpath) as f:c=f.read()
 old1='            setup_3dgut(conf)\n            import lib3dgut_cc as tdgut  # type: ignore'
 new1='            setup_3dgut(conf)\n            import sys, os, glob\n            cache_pattern = os.path.expanduser("~/.cache/torch_extensions/*/lib3dgut_cc/lib3dgut_cc*.so")\n            so_files = glob.glob(cache_pattern)\n            if so_files and os.path.dirname(so_files[0]) not in sys.path:\n                sys.path.insert(0, os.path.dirname(so_files[0]))\n            import lib3dgut_cc as tdgut  # type: ignore'
 old2='            setup_3dgrt(conf)\n            import lib3dgrt_cc as tdgrt  # type: ignore'
 new2='            setup_3dgrt(conf)\n            import sys, os, glob\n            cache_pattern = os.path.expanduser("~/.cache/torch_extensions/*/lib3dgrt_cc/lib3dgrt_cc*.so")\n            so_files = glob.glob(cache_pattern)\n            if so_files and os.path.dirname(so_files[0]) not in sys.path:\n                sys.path.insert(0, os.path.dirname(so_files[0]))\n            import lib3dgrt_cc as tdgrt  # type: ignore'
 old3='        import lib_mcmc_cc as gaussian_mcmc'
 new3='        import sys, os, glob\n        cache_pattern = os.path.expanduser("~/.cache/torch_extensions/*/lib_mcmc_cc/lib_mcmc_cc*.so")\n        so_files = glob.glob(cache_pattern)\n        if so_files and os.path.dirname(so_files[0]) not in sys.path:\n            sys.path.insert(0, os.path.dirname(so_files[0]))\n        import lib_mcmc_cc as gaussian_mcmc'
 old4='        import lib_optimizers_cc as optimizers_cc'
 new4='        import sys, os, glob\n        cache_pattern = os.path.expanduser("~/.cache/torch_extensions/*/lib_optimizers_cc/lib_optimizers_cc*.so")\n        so_files = glob.glob(cache_pattern)\n        if so_files and os.path.dirname(so_files[0]) not in sys.path:\n            sys.path.insert(0, os.path.dirname(so_files[0]))\n        import lib_optimizers_cc as optimizers_cc'
 c=c.replace(old1,new1).replace(old2,new2).replace(old3,new3).replace(old4,new4)
 with open(fpath,'w') as f:f.write(c)
