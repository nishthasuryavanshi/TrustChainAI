"""Smoke tests: every page renders without exceptions against the bootstrapped demo DB."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config as cfg

PAGES = ["app.py", "pages/1_ML_Risk_Prediction.py", "pages/2_Product_Traceability.py",
         "pages/3_Blockchain_Verification.py", "pages/4_Investigation_Center.py"]
pytestmark = pytest.mark.skipif(not cfg.DB_PATH.exists(), reason="run `python -m src.bootstrap` first")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(cfg.ROOT / page), default_timeout=60).run()
    assert not at.exception, at.exception
