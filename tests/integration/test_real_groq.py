import os

import pytest

from .real_llm_cases import CASE_NAMES, run_case

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_LLM_TESTS") != "1",
        reason="Set RUN_REAL_LLM_TESTS=1 to call the real Groq API",
    ),
]


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_real_groq_case(case_name):
    result = run_case(case_name)
    assert result.passed, f"{result.name} failed: {result.error}\nDetails: {result.details}"
