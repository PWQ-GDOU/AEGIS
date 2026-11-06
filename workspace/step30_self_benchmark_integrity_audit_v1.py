"""Fail-closed integrity audit for participant-built benchmark score reports."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Mapping, Sequence
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file
VERSION = "step30-self-benchmark-integrity-audit-v1"
def _read(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise ValueError("benchmark report must be an object")
    return dict(value)
def audit_benchmark(path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path).resolve(); report = _read(target); findings = []; eligible = True; version = str(report.get("version", ""))
    if report.get("official_accuracy_gate") is not False or report.get("official_reference_gate") is not False:
        eligible = False; findings.append({"severity":"error","code":"official_gate_semantics_changed","reason":"participant-built benchmark may not open an official gate"})
    if report.get("fem_gate") != "blocked":
        eligible = False; findings.append({"severity":"error","code":"fem_gate_semantics_changed","reason":"participant-built benchmark may not open FEM gate"})
    if "v2-model-combination" in version:
        eligible = False; findings.append({"severity":"error","code":"component_leakage_in_combination_lomo","reason":"combined candidate includes every model while each reference omits one"})
    elif report.get("reference_independence") == "leave_one_model_out_and_leave_one_seed_out":
        findings.append({"severity":"warning","code":"cross_model_consistency_not_measured_truth","reason":"conditional internal consistency only until independent baseline/worn depth is supplied"})
    if report.get("measured_truth") is not True:
        findings.append({"severity":"warning","code":"no_independent_measured_truth","reason":"self-built reference cannot be reported as physical spatial accuracy"})
    score = report.get("competition_self_benchmark_score")
    if not isinstance(score, (int,float)):
        eligible = False; findings.append({"severity":"error","code":"score_missing","reason":"numeric conditional score is missing"})
    result = {"version":VERSION,"status":"eligible_conditional_internal_consistency_only" if eligible else "blocked_for_independent_combination_score","source_report":{"path":str(target),"sha256":sha256_file(target),"version":version},"conditional_score_value":float(score) if isinstance(score,(int,float)) else None,"competition_score_claim_allowed":False,"conditional_internal_consistency_claim_allowed":bool(eligible),"official_accuracy_gate":False,"official_reference_gate":False,"fem_gate":"blocked","findings":findings,"required_next_evidence":["independent participant-built or measured baseline/worn depth field","reference-to-DEC-face registration","frozen same-material amplitude calibration"]}
    result["report_sha256"] = canonical_sha256(result)
    if output_path is not None: safe_json_dump(result, Path(output_path).resolve())
    return result
def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(argv); result=audit_benchmark(args.input,output_path=args.output); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
