from __future__ import annotations
import re
from collections import defaultdict

import pytest
from _pytest.config import Config
from _pytest.runner import CallInfo, TestReport

# Section mapping                                                             #
SECTION_PREFIX = {
    "FD":  "data",        # Features & Data
    "ML":  "model",       # Model Development
    "INF": "infra",       # Infrastructure
    "MON": "monitoring",  # Monitoring
}
SECTIONS            = tuple(SECTION_PREFIX.values())
POINTS_PER_ID       = 1          
TESTS_PER_SECTION   = 7          



def _section_from_id(test_id: str) -> str:
    """Map 'FD-1', 'ML-6', … → section name; fallback 'unknown'."""
    m = re.match(r"([A-Z]+)-", test_id)
    return SECTION_PREFIX.get(m.group(1), "unknown") if m else "unknown"


# Pytest hooks                                                                #
def pytest_configure(config: Config) -> None:
    """Runs once, early.  Register marker & create tracker."""
    config.addinivalue_line(
        "markers",
        "ml_test(id): mark test with a Google-ML-Test-Score identifier",
    )
    config._section_tracker: dict[str, set[str]] = defaultdict(set)


@pytest.hookimpl(hookwrapper=True)   
def pytest_runtest_makereport(item, call: CallInfo) -> TestReport:
    """
    Runs after each test phase (setup/call/teardown). We wait for Pytest to
    create the TestReport (`yield`), then look only at the *call* phase that
    passed, grab the marker, and record the ID.
    """
    outcome = yield
    report: TestReport = outcome.get_result()

    if report.when != "call" or report.outcome != "passed":
        return

    mark = item.get_closest_marker("ml_test")
    if not mark:
        return

    test_id = mark.args[0]
    section = _section_from_id(test_id)
    item.config._section_tracker[section].add(test_id)


def pytest_sessionfinish(session, exitstatus: int) -> None:
    """At the very end: compute section scores and print the summary."""
    tracker = session.config._section_tracker
    section_scores = {sec: len(tracker.get(sec, set())) * POINTS_PER_ID
                      for sec in SECTIONS}
    final_score = min(section_scores.values())         

    print("\n===== ML TEST SCORE =====")
    for sec in SECTIONS:
        print(f" {sec.capitalize():<12}: {section_scores[sec]}/{TESTS_PER_SECTION}")
    print(" -------------------------")
    print(f" Final score   : {final_score}/{TESTS_PER_SECTION}")
    print("================================\n")
