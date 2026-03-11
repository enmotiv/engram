"""Unit tests for middleware: error handlers, HTTP status mapping."""

import pytest


class TestHttpStatusToCode:
    def test_known_codes(self):
        from engram.core.middleware import _http_status_to_code

        assert _http_status_to_code(400) == "INVALID_INPUT"
        assert _http_status_to_code(401) == "UNAUTHORIZED"
        assert _http_status_to_code(403) == "FORBIDDEN"
        assert _http_status_to_code(404) == "NOT_FOUND"
        assert _http_status_to_code(405) == "METHOD_NOT_ALLOWED"
        assert _http_status_to_code(409) == "CONFLICT"
        assert _http_status_to_code(429) == "RATE_LIMITED"
        assert _http_status_to_code(500) == "INTERNAL_ERROR"
        assert _http_status_to_code(503) == "SERVICE_UNAVAILABLE"

    def test_unknown_code(self):
        from engram.core.middleware import _http_status_to_code

        assert _http_status_to_code(418) == "HTTP_418"
        assert _http_status_to_code(502) == "HTTP_502"


class TestHttpCodeMap:
    def test_map_completeness(self):
        from engram.core.middleware import _HTTP_CODE_MAP

        assert isinstance(_HTTP_CODE_MAP, dict)
        assert len(_HTTP_CODE_MAP) == 9
