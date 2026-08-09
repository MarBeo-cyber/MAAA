"""The REST API in api/maaa_api.py — documented nowhere before, tested by nothing.

docs/API_SPEC.md described a JSON API over maaa_core that does not exist, while
this Flask app (the one that does) had no spec and no tests.
"""

from __future__ import annotations

import pytest

import maaa_config
from api.maaa_api import create_maaa_app
from core.maaa_agent import MAAAAgent


@pytest.fixture()
def client(tmp_path):
    agent = MAAAAgent(simulation_mode=True, verbose=False,
                      db_path=str(tmp_path / "e.db"),
                      autobio_path=str(tmp_path / "a.json"))
    app = create_maaa_app(agent)
    app.config.update(TESTING=True)
    with app.test_client() as c:
        c.post("/tick/n/5")
        yield c
    agent.l5.close()


GET_ENDPOINTS = ["/status", "/snapshot", "/human", "/risk", "/guidance",
                 "/memory/working", "/memory/episodic", "/memory/recall",
                 "/session", "/override"]


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_get_endpoints_return_json(client, path):
    resp = client.get(path)
    assert resp.status_code == 200, (path, resp.status_code)
    assert resp.is_json
    assert isinstance(resp.get_json(), dict)


def test_status_reports_the_unified_version_and_synthetic_source(client):
    body = client.get("/status").get_json()
    assert body["agent"] == f"MAAA v{maaa_config.VERSION}"
    assert body["data_source"] == "synthetic"
    assert body["simulation_mode"] is True


def test_status_does_not_fabricate_a_battery_reading(client):
    health = client.get("/status").get_json()["last"]["health"]
    assert health["battery_pct"] is None
    assert health["battery_source"] == "unavailable"


@pytest.mark.parametrize("scenario", ["normal", "smoky", "dark", "dusty",
                                      "obstructed", "collapsed", "panic", "frozen"])
def test_scenario_injection(client, scenario):
    resp = client.post(f"/scenario/{scenario}")
    assert resp.status_code == 200
    assert resp.get_json() == {"scenario": scenario, "injected": True}


def test_unknown_scenario_is_rejected_with_the_available_list(client):
    resp = client.post("/scenario/bogus")
    assert resp.status_code == 400
    body = resp.get_json()
    assert "available" in body and "normal" in body["available"]


def test_override_endpoint_round_trip(client):
    assert client.get("/override").get_json()["override"] is None
    assert client.post("/override", json={"command": "mute"}).status_code == 200
    assert client.get("/override").get_json()["override"] == "mute"
    assert client.post("/override", json={"command": "resume"}).status_code == 200
    assert client.get("/override").get_json()["override"] is None


def test_unknown_override_is_rejected(client):
    resp = client.post("/override", json={"command": "launch"})
    assert resp.status_code == 400
    assert "available" in resp.get_json()


def test_tick_endpoint_advances_the_pipeline(client):
    before = client.get("/status").get_json()["tick"]
    client.post("/tick/n/3")
    after = client.get("/status").get_json()["tick"]
    assert after == before + 3
