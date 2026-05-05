from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from model_loader import predict, get_available_drugs
import dicttoxml
import time
import torch
import os

app = FastAPI(title="DDI Graph Model API")

class DDIRequest(BaseModel):
    drugA: str
    drugB: str

class DrugInfo(BaseModel):
    drugId: str
    name: str

class AddDrugRequest(BaseModel):
    name: str
    smiles: str

def risk_label(p: float) -> str:
    if p < 0.3:
        return "Low"
    if p < 0.6:
        return "Medium"
    return "High"

# 1. Lightweight health-check endpoint (Use this for Render keep-alive)
@app.get(
    "/health", 
    tags=["Health"], 
    summary="Perform a Health Check", 
    status_code=status.HTTP_200_OK
)
def health_check():
    return {"status": "healthy"}

@app.get("/status", tags=["Health"], summary="Full production readiness check")
def status_check():
    checks = {}
    overall = "ready"

    # 1. Model artifacts
    try:
        from model_loader import node_embeddings, decoder, drug_to_node, smiles_dict
        checks["node_embeddings"] = {
            "ok": True,
            "detail": f"{node_embeddings.shape[0]} nodes, dim={node_embeddings.shape[1]}"
        }
        checks["edge_decoder"] = {
            "ok": True,
            "detail": f"{sum(p.numel() for p in decoder.parameters())} parameters"
        }
        checks["drug_index"] = {
            "ok": True,
            "detail": f"{len(drug_to_node)} drugs with node embeddings, {len(smiles_dict)} with SMILES"
        }
    except Exception as e:
        checks["model_artifacts"] = {"ok": False, "detail": str(e)}
        overall = "degraded"

    # 2. Smoke prediction (no network call)
    try:
        from model_loader import drug_to_node
        ids = list(drug_to_node.keys())[:2]
        t0 = time.perf_counter()
        result = predict(ids[0], ids[1], include_properties=False)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        prob = result["probability"]
        checks["inference"] = {
            "ok": True,
            "detail": f"predict({ids[0]}, {ids[1]}) = {prob:.4f} in {latency_ms}ms"
        }
    except Exception as e:
        checks["inference"] = {"ok": False, "detail": str(e)}
        overall = "not_ready"

    # 3. Drug name lookup
    try:
        from model_loader import drugid_to_name
        checks["drug_names"] = {
            "ok": True,
            "detail": f"{len(drugid_to_name)} named drugs loaded"
        }
    except Exception as e:
        checks["drug_names"] = {"ok": False, "detail": str(e)}
        overall = "degraded"

    # 4. Hardware
    checks["hardware"] = {
        "ok": True,
        "detail": f"torch={torch.__version__}, device=cpu, cuda_available={torch.cuda.is_available()}"
    }

    http_status = status.HTTP_200_OK if overall == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=http_status,
        content={"status": overall, "checks": checks}
    )

@app.get("/drugs", response_model=list[DrugInfo])
def get_drugs():
    drugs = get_available_drugs()
    return drugs

def _validate_smiles(smiles: str) -> bool:
    if not smiles or not isinstance(smiles, str):
        return False
    return True

@app.post("/add-drug")
def add_drug(req: AddDrugRequest):
    from model_loader import smiles_dict, drugid_to_name
    if not _validate_smiles(req.smiles):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SMILES: {req.smiles}. Please check the SMILES string format."
        )
    existing_ids = set(smiles_dict.keys())
    idx = 1
    while True:
        new_id = f"DB{idx:05d}"
        if new_id not in existing_ids:
            break
        idx += 1
    smiles_dict[new_id] = req.smiles
    drugid_to_name[new_id] = req.name
    return {"drugId": new_id, "name": req.name, "smiles": req.smiles}

@app.post("/predict")
def predict_ddi(req: DDIRequest):
    try:
        result = predict(req.drugA, req.drugB, include_properties=False)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "drugA": req.drugA,
        "drugB": req.drugB,
        "probability": result["probability"],
        "risk": risk_label(result["probability"]),
        "embedding_similarity": result["embedding_similarity"],
        "drug_a_smiles": result["drug_a_smiles"],
        "drug_b_smiles": result["drug_b_smiles"],
    }

@app.post("/predict/extended")
def predict_ddi_extended(req: DDIRequest):
    try:
        result = predict(req.drugA, req.drugB, include_properties=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    from model_loader import drugid_to_name
    drug_a_name = drugid_to_name.get(req.drugA, req.drugA)
    drug_b_name = drugid_to_name.get(req.drugB, req.drugB)
    return {
        "drugA": req.drugA,
        "drugA_name": drug_a_name,
        "drugB": req.drugB,
        "drugB_name": drug_b_name,
        "probability": result["probability"],
        "risk": risk_label(result["probability"]),
        "embedding_similarity": result["embedding_similarity"],
        "drug_a_properties": result["drug_a_properties"],
        "drug_b_properties": result["drug_b_properties"],
        "drug_a_smiles": result["drug_a_smiles"],
        "drug_b_smiles": result["drug_b_smiles"],
    }

@app.post("/predict/xml")
def predict_xml(req: DDIRequest):
    try:
        result = predict(req.drugA, req.drugB, include_properties=False)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    payload = {
        "drugA": req.drugA,
        "drugB": req.drugB,
        "probability": result["probability"],
        "risk": risk_label(result["probability"]),
        "embedding_similarity": result["embedding_similarity"],
        "drug_a_smiles": result["drug_a_smiles"],
        "drug_b_smiles": result["drug_b_smiles"],
    }
    xml_data = dicttoxml.dicttoxml(payload, custom_root="DDIResponse", attr_type=False)
    return Response(content=xml_data, media_type="application/xml")

@app.get("/test-report", tags=["Health"], summary="Run test suite and return results")
def test_report():
    import pytest

    class _Collector:
        def __init__(self):
            self.passed = []
            self.failed = []
            self.errors = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    self.passed.append(report.nodeid)
                elif report.failed:
                    self.failed.append({
                        "test": report.nodeid,
                        "message": str(report.longrepr).splitlines()[-1]
                    })
            elif report.when == "setup" and report.failed:
                self.errors.append({
                    "test": report.nodeid,
                    "message": str(report.longrepr).splitlines()[-1]
                })

    collector = _Collector()
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    exit_code = pytest.main(
        [tests_dir, "--tb=short", "-q", "--no-header", "-p", "no:cacheprovider"],
        plugins=[collector]
    )

    total = len(collector.passed) + len(collector.failed) + len(collector.errors)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200 if exit_code == 0 else 207,
        content={
            "status": "pass" if exit_code == 0 else "fail",
            "summary": {
                "passed": len(collector.passed),
                "failed": len(collector.failed),
                "errors": len(collector.errors),
                "total": total
            },
            "passed": collector.passed,
            "failed": collector.failed,
            "errors": collector.errors
        }
    )