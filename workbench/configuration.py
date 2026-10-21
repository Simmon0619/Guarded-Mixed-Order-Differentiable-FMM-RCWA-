"""Versioned JSON input for batch runs using the existing workbench parameters."""
import copy
import json
import math
from pathlib import Path

PARAMETERS = set("""
accept_rule beta_ladder cd_bias_nm chain_alphas chain_beta chain_jumps chain_noise_seed
chain_redescent_steps chain_start ckpt_every compare_lumerical compare_restart_budget
compare_torcwa compare_torcwa_opt engine field_views field_zin_nm field_zmax_nm filter_nm
gm_beta0 gm_beta_end gm_cadence gm_guards gm_mix_from_beta gm_mix_k gm_rise jx_im jx_re
jy_im jy_re level_descent level_gate levels_certify lr0 lumerical_kvecs lx_nm ly_nm
memory_efficient n_background n_feature n_incident n_output nx ny pattern_source phi_deg
polarization protocol repair_bar repair_beta_fixed repair_beta_min repair_beta_mode
repair_enabled repair_hops repair_optimism_max repair_re_k repair_stop_after repair_trigger
save_on_cpu seeds steps symmetry target theta_deg thickness_nm threshold wavelength_nm
""".split())


def load_config(path):
    def invalid_constant(value):
        raise ValueError("non-finite JSON number: " + value)
    with Path(path).open(encoding="utf-8-sig") as stream:
        return validate_config(json.load(stream, parse_constant=invalid_constant))


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(config) - {"schema_version", "name", "description", "type", "params"}
    if unknown:
        raise ValueError("unknown configuration keys: " + ", ".join(sorted(unknown)))
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if config.get("type") not in ("solve", "optimize"):
        raise ValueError("type must be solve or optimize")
    p = config.get("params")
    if not isinstance(p, dict):
        raise ValueError("params must be an object")
    unknown = set(p) - PARAMETERS
    if unknown:
        raise ValueError("unknown params: " + ", ".join(sorted(unknown)))

    def finite_tree(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("configuration contains a non-finite number")
        if isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite_tree(item)
    finite_tree(p)

    def positive(key, integer=False, zero=False):
        if key not in p:
            return
        value = p[key]
        valid_type = type(value) is int if integer else type(value) in (int, float)
        if not valid_type or (value < 0 if zero else value <= 0):
            raise ValueError(key + " must be a " + ("non-negative" if zero else "positive") + (" integer" if integer else " number"))
    for key in ("nx", "ny", "steps", "gm_mix_k", "ckpt_every", "chain_redescent_steps", "repair_re_k", "repair_stop_after"):
        positive(key, integer=True)
    for key in ("chain_jumps", "repair_hops"):
        positive(key, integer=True, zero=True)
    for key in ("wavelength_nm", "lx_nm", "ly_nm", "filter_nm", "n_incident", "n_output", "n_feature", "n_background",
                "lr0", "gm_beta0", "gm_beta_end", "gm_mix_from_beta", "chain_beta", "repair_beta_fixed", "repair_beta_min"):
        positive(key)
    positive("thickness_nm", zero=True)
    if "threshold" in p and (type(p["threshold"]) not in (int, float) or not 0 < p["threshold"] < 1):
        raise ValueError("threshold must lie strictly between 0 and 1")
    for key, choices in {"engine": ("cpu_c128", "gpu_c64"), "symmetry": ("none", "x", "y", "xy"),
                         "polarization": ("x", "y", "custom"), "protocol": ("guarded_mixed", "legacy_chain", "chain"),
                         "accept_rule": ("nominal", "maximin_cd"), "repair_trigger": ("manual", "always", "numeric"),
                         "repair_beta_mode": ("fixed", "star")}.items():
        if key in p and p[key] not in choices:
            raise ValueError(key + " must be one of " + ", ".join(choices))
    for key in ("gm_guards", "gm_cadence", "gm_rise", "repair_enabled", "memory_efficient",
                "compare_torcwa", "compare_lumerical", "compare_restart_budget", "compare_torcwa_opt"):
        if key in p and type(p[key]) is not bool:
            raise ValueError(key + " must be a boolean")
    def pair(value, label, nonnegative=True):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int or (nonnegative and v < 0) for v in value):
            raise ValueError(label + " must contain two " + ("non-negative " if nonnegative else "") + "integers")
    for key in ("level_descent", "level_gate", "target"):
        if key in p:
            pair(p[key], key, key != "target")
    if "levels_certify" in p:
        if not isinstance(p["levels_certify"], list):
            raise ValueError("levels_certify must be a list of order pairs")
        for lv in p["levels_certify"]:
            pair(lv, "levels_certify entry")
    if "seeds" in p and (not isinstance(p["seeds"], list) or not p["seeds"] or any(type(s) is not int for s in p["seeds"])):
        raise ValueError("seeds must be a nonempty list of integers")
    if "pattern_source" in p:
        source = p["pattern_source"]
        if not isinstance(source, dict) or source.get("kind") not in ("seed", "uniform", "run_best"):
            raise ValueError("pattern_source.kind must be seed, uniform or run_best")
        if source["kind"] == "uniform" and (type(source.get("fill", 1.0)) not in (int, float) or not 0 <= source.get("fill", 1.0) <= 1):
            raise ValueError("pattern_source.fill must lie between 0 and 1")
        if source["kind"] == "run_best" and not source.get("run_id"):
            raise ValueError("pattern_source.run_id is required for run_best")
    return copy.deepcopy(config)
