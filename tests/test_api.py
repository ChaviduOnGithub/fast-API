"""
Integration tests for the DDI FastAPI endpoints.
Run with: pytest fast-API/tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
from inference_api import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures — two known drug IDs that are present in node_embeddings
# ---------------------------------------------------------------------------
DRUG_A = "DB00001"
DRUG_B = "DB00002"
UNKNOWN_DRUG = "DB99999"


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self):
        r = client.get("/health")
        assert r.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# 2. GET /status
# ---------------------------------------------------------------------------
class TestStatus:
    def test_returns_200_when_ready(self):
        r = client.get("/status")
        assert r.status_code == 200

    def test_status_is_ready(self):
        r = client.get("/status")
        assert r.json()["status"] == "ready"

    def test_all_checks_pass(self):
        r = client.get("/status")
        checks = r.json()["checks"]
        for name, result in checks.items():
            assert result["ok"] is True, f"Check '{name}' failed: {result['detail']}"

    def test_inference_check_present(self):
        r = client.get("/status")
        assert "inference" in r.json()["checks"]

    def test_hardware_check_present(self):
        r = client.get("/status")
        assert "hardware" in r.json()["checks"]


# ---------------------------------------------------------------------------
# 2. GET /drugs
# ---------------------------------------------------------------------------
class TestGetDrugs:
    def test_returns_200(self):
        r = client.get("/drugs")
        assert r.status_code == 200

    def test_returns_list(self):
        r = client.get("/drugs")
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_drug_schema(self):
        r = client.get("/drugs")
        first = r.json()[0]
        assert "drugId" in first
        assert "name" in first


# ---------------------------------------------------------------------------
# 3. POST /add-drug
# ---------------------------------------------------------------------------
class TestAddDrug:
    def test_add_valid_drug(self):
        r = client.post("/add-drug", json={"name": "TestDrug", "smiles": "CCO"})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "TestDrug"
        assert body["smiles"] == "CCO"
        assert body["drugId"].startswith("DB")

    def test_add_drug_appears_in_drugs_list(self):
        r = client.post("/add-drug", json={"name": "UniqueTestDrug", "smiles": "C1CCCCC1"})
        new_id = r.json()["drugId"]
        drugs = client.get("/drugs").json()
        ids = [d["drugId"] for d in drugs]
        assert new_id in ids

    def test_add_drug_empty_smiles_returns_400(self):
        r = client.post("/add-drug", json={"name": "BadDrug", "smiles": ""})
        assert r.status_code == 400

    def test_add_drug_missing_fields_returns_422(self):
        r = client.post("/add-drug", json={"name": "OnlyName"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 4. POST /predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_valid_pair_returns_200(self):
        r = client.post("/predict", json={"drugA": DRUG_A, "drugB": DRUG_B})
        assert r.status_code == 200

    def test_response_fields(self):
        r = client.post("/predict", json={"drugA": DRUG_A, "drugB": DRUG_B})
        body = r.json()
        for key in ("drugA", "drugB", "probability", "risk", "embedding_similarity"):
            assert key in body, f"Missing key: {key}"

    def test_probability_in_range(self):
        r = client.post("/predict", json={"drugA": DRUG_A, "drugB": DRUG_B})
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_label_valid(self):
        r = client.post("/predict", json={"drugA": DRUG_A, "drugB": DRUG_B})
        assert r.json()["risk"] in ("Low", "Medium", "High")

    def test_unknown_drug_returns_404(self):
        r = client.post("/predict", json={"drugA": UNKNOWN_DRUG, "drugB": DRUG_B})
        assert r.status_code == 404

    def test_both_unknown_returns_404(self):
        r = client.post("/predict", json={"drugA": UNKNOWN_DRUG, "drugB": "DB88888"})
        assert r.status_code == 404

    def test_missing_field_returns_422(self):
        r = client.post("/predict", json={"drugA": DRUG_A})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 5. POST /predict/xml
# ---------------------------------------------------------------------------
class TestPredictXML:
    def test_returns_200(self):
        r = client.post("/predict/xml", json={"drugA": DRUG_A, "drugB": DRUG_B})
        assert r.status_code == 200

    def test_content_type_is_xml(self):
        r = client.post("/predict/xml", json={"drugA": DRUG_A, "drugB": DRUG_B})
        assert "application/xml" in r.headers["content-type"]

    def test_body_is_parseable_xml(self):
        import xml.etree.ElementTree as ET
        r = client.post("/predict/xml", json={"drugA": DRUG_A, "drugB": DRUG_B})
        root = ET.fromstring(r.content)   # raises if invalid XML
        assert root.tag == "DDIResponse"

    def test_xml_contains_probability(self):
        import xml.etree.ElementTree as ET
        r = client.post("/predict/xml", json={"drugA": DRUG_A, "drugB": DRUG_B})
        root = ET.fromstring(r.content)
        prob_elem = root.find("probability")
        assert prob_elem is not None
        assert 0.0 <= float(prob_elem.text) <= 1.0

    def test_unknown_drug_returns_404(self):
        r = client.post("/predict/xml", json={"drugA": UNKNOWN_DRUG, "drugB": DRUG_B})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 6. POST /predict/extended
# ---------------------------------------------------------------------------
class TestPredictExtended:
    def test_returns_200(self):
        r = client.post("/predict/extended", json={"drugA": DRUG_A, "drugB": DRUG_B})
        assert r.status_code == 200

    def test_extended_fields_present(self):
        r = client.post("/predict/extended", json={"drugA": DRUG_A, "drugB": DRUG_B})
        body = r.json()
        for key in ("drugA_name", "drugB_name", "probability", "risk"):
            assert key in body, f"Missing key: {key}"

    def test_unknown_drug_returns_404(self):
        r = client.post("/predict/extended", json={"drugA": UNKNOWN_DRUG, "drugB": DRUG_B})
        assert r.status_code == 404
