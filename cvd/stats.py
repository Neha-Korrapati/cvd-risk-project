"""Paper Sec. II-F: Kruskal-Wallis with post-hoc Dunn's test (Bonferroni) when
comparing several models, Mann-Whitney U when comparing two."""
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal, mannwhitneyu


def stars(p: float) -> str:
    # Significance levels used in Fig. 3 and Fig. 6.
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def kruskal_dunn(scores: dict[str, list[float]], reference: str) -> tuple[float, pd.DataFrame]:
    """Kruskal-Wallis across all groups, then Dunn/Bonferroni p-values of each
    group vs `reference`. Returns (kruskal_p, table)."""
    names = list(scores)
    _, kw_p = kruskal(*[scores[n] for n in names])
    long = pd.DataFrame([(n, v) for n in names for v in scores[n]], columns=["group", "value"])
    dunn = sp.posthoc_dunn(long, val_col="value", group_col="group", p_adjust="bonferroni")
    rows = [{"group": n, "p_vs_reference": float(dunn.loc[n, reference]),
             "sig": stars(float(dunn.loc[n, reference]))}
            for n in names if n != reference]
    return float(kw_p), pd.DataFrame(rows)


def mann_whitney(a: list[float], b: list[float]) -> float:
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
