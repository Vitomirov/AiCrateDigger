"""Human-readable and JSON reporting for evaluation runs."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from eval.schema import EvalReport


def print_report(report: EvalReport, *, stream: TextIO | None = None) -> None:
    out = stream or sys.stdout
    out.write("\n")
    out.write(f"AiCrateDigger pipeline evaluation — dataset v{report.dataset_version}\n")
    out.write(f"Mode filter: {report.mode}\n")
    out.write(f"Cases: {report.total}  Passed: {report.passed}  Failed: {report.failed}\n")
    out.write("-" * 72 + "\n")

    for case in report.cases:
        mark = "PASS" if case.passed else "FAIL"
        out.write(f"[{mark}] {case.case_id}\n")
        out.write(f"       query: {case.query[:96]}{'…' if len(case.query) > 96 else ''}\n")
        if case.reason is not None:
            out.write(f"       reason: {case.reason}\n")
        if case.result_count is not None:
            out.write(f"       results: {case.result_count}\n")
        for stage in case.stages:
            if stage.skipped:
                continue
            smark = "ok" if stage.passed else "!!"
            status = stage.status or "—"
            out.write(f"       stage {stage.name}: {smark} ({status})\n")
        for err in case.errors:
            out.write(f"       error: {err}\n")
        out.write("\n")

    out.write("-" * 72 + "\n")
    if report.failed:
        out.write(f"FAILED ({report.failed} case(s))\n")
    else:
        out.write("ALL CASES PASSED\n")


def report_to_json(report: EvalReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
