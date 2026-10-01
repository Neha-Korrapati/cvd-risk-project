"""Reproduction of Vyshnya et al., "Optimized Clinical Feature Analysis for
Improved Cardiovascular Disease Risk Screening", IEEE OJEMB vol. 5, 2024."""
import os
import warnings

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# Expected, harmless warnings under the paper's settings: SVC is capped at
# max_iter=100 (Appendix B), and KNN grid values of k larger than an inner
# training fold fail and are scored NaN. Silenced here and in worker processes.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite")
# scikit-learn 1.8+ deprecates the `penalty=` argument; we keep it because
# Appendix B specifies the penalty explicitly, and it still works in 1.9.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
