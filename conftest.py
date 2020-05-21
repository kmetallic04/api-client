# contents of test_app.py, a simple test for our API retrieval
import pytest
import requests


# Prevent `requests` from making any inadvertent API calls in tests
@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    monkeypatch.delattr("requests.sessions.Session.request")


class MockResponse(requests.Response):
    def __init__(self, status_code, response, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_code = status_code
        self.mock_response = response

    @staticmethod
    def json():
        return mock_response

# monkeypatched requests.get moved to a fixture
@pytest.fixture(autouse=True)
def mock_response(monkeypatch):
    """Requests.get() mocked to return {'mock_key':'mock_response'}."""

    def mock_get(*args, **kwargs):
        # TODO: detect what request is being made and return corresponding mock data
        url = args[0]
        params = kwargs['params']
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)