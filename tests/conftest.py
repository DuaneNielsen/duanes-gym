# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-render", action="store_true", default="False", help="my option: type1 or type2"
    )


@pytest.fixture
def render(request):
    return not request.config.getoption("--no-render")
