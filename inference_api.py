from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import predict, get_available_drugs

app = FastAPI(title="DDI Graph Model API")

class DDIRequest(BaseModel):
    drugA: str
    drugB: str

class MolecularProperties(BaseModel):
    molecular_weight: float
    num_atoms: int
    num_bonds: int
    logp: float
    num_h_donors: int
    num_h_acceptors: int

class DDIResponse(BaseModel):
    drugA: str
    drugB: str
    probability: float
    risk: str
    embedding_similarity: float
    drug_a_properties: MolecularProperties
    drug_b_properties: MolecularProperties
    drug_a_smiles: str
    drug_b_smiles: str
    drug_a_neighbors: list[str]
    drug_b_neighbors: list[str]

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

@app.get("/drugs", response_model=list[DrugInfo])
def get_drugs():
    drugs = get_available_drugs()
    return drugs

def _validate_smiles(smiles: str) -> bool:
    if not smiles or not isinstance(smiles, str):
        return False
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

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

@app.post("/predict", response_model=DDIResponse)
def predict_ddi(req: DDIRequest):
    try:
        result = predict(req.drugA, req.drugB)
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
        "drug_a_neighbors": result["drug_a_neighbors"],
        "drug_b_neighbors": result["drug_b_neighbors"]
    }
