import torch
import pandas as pd
from models import EdgeDecoder
import os
import numpy as np
import requests

DEVICE = torch.device("cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "artifacts_latest")

node_embeddings = torch.load(f"{ARTIFACT_DIR}/node_embeddings.pt", map_location=DEVICE)
node_embeddings.requires_grad_(False)
hidden_dim = node_embeddings.shape[1]
decoder = EdgeDecoder(hidden_dim)
decoder.load_state_dict(torch.load(f"{ARTIFACT_DIR}/edge_decoder.pt", map_location=DEVICE))
decoder.eval()
df = pd.read_csv(f"{ARTIFACT_DIR}/nodeidx2drugid_with_smiles.csv.gz", compression="gzip")
import_path = os.path.join(SCRIPT_DIR, "data", "unique_drugid_name.csv")
if os.path.exists(import_path):
    drug_name_df = pd.read_csv(import_path)
    drugid_to_name = dict(zip(drug_name_df["drug_id"], drug_name_df["drug_name"]))
else:
    drugid_to_name = {}
drug_to_node = dict(zip(df["drug id"], df["node idx"]))
node_to_drug = dict(zip(df["node idx"], df["drug id"]))
smiles_dict = dict(zip(df["drug id"], df["smiles"]))
def get_available_drugs() -> list:
    drugs_dict = {}
    for _, row in df.iterrows():
        drug_id = row["drug id"]
        drugs_dict[drug_id] = {
            "drugId": drug_id,
            "name": drugid_to_name.get(drug_id, drug_id)
        }
    for drug_id in smiles_dict.keys():
        if drug_id not in drugs_dict:
            drugs_dict[drug_id] = {
                "drugId": drug_id,
                "name": drugid_to_name.get(drug_id, drug_id)
            }
    return list(drugs_dict.values())
def _extract_molecular_properties(smiles: str) -> dict:
    import logging
    import xml.etree.ElementTree as ET
    if not smiles:
        logging.warning("No SMILES provided for property extraction.")
        return {}
    # Step 1: Get CID from PubChem
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    try:
        cid_resp = requests.get(cid_url, timeout=5)
        cid_resp.raise_for_status()
        cid_data = cid_resp.json()
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            logging.warning(f"PubChem could not find CID for SMILES: {smiles}")
            return {}
        cid = cids[0]
    except Exception as e:
        logging.error(f"PubChem CID lookup failed for SMILES: {smiles} | Error: {e}")
        return {}
    # Step 2: Try JSON compound endpoint
    compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/JSON"
    try:
        compound_resp = requests.get(compound_url, timeout=5)
        compound_resp.raise_for_status()
        compound_data = compound_resp.json()
        props = {}
        
        # Extract from PC_Compounds[0]
        if "PC_Compounds" in compound_data and len(compound_data["PC_Compounds"]) > 0:
            compound = compound_data["PC_Compounds"][0]
            
            # Extract compound properties
            if "props" in compound:
                for prop in compound["props"]:
                    if "urn" in prop and "value" in prop:
                        key = prop["urn"].get("label", "Unknown")
                        val = prop["value"]
                        # Extract value from nested structure
                        if isinstance(val, dict):
                            if "sval" in val:
                                props[key] = val["sval"]
                            elif "ival" in val:
                                props[key] = val["ival"]
                            elif "fval" in val:
                                props[key] = val["fval"]
                        else:
                            props[key] = val
            
            logging.info(f"Extracted {len(props)} properties from JSON for CID: {cid}")
            if props:
                return props
    except Exception as e:
        logging.warning(f"PubChem JSON compound lookup failed for CID: {cid} | Error: {e}")
    # Step 3: Fallback to XML endpoint
    prop_url_xml = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/XML"
    try:
        xml_resp = requests.get(prop_url_xml, timeout=5)
        xml_resp.raise_for_status()
        xml_root = ET.fromstring(xml_resp.content)
        props = {}
        
        # Handle namespaces
        ns = {'pc': 'http://www.ncbi.nlm.nih.gov/pug'}
        
        # Find PC-Compound (with or without namespace)
        compound = xml_root.find(".//pc:PC-Compound", ns)
        if compound is None:
            compound = xml_root.find(".//PC-Compound")
        
        if compound is not None:
            # Look for compound properties
            props_elem = compound.find(".//pc:PC-Compound_props", ns)
            if props_elem is None:
                props_elem = compound.find(".//PC-Compound_props")
            
            if props_elem is not None:
                # Find all properties
                prop_list = props_elem.findall(".//pc:PC-CompoundProperty", ns)
                if not prop_list:
                    prop_list = props_elem.findall(".//PC-CompoundProperty")
                
                for prop in prop_list:
                    # Get property urn/label
                    urn = prop.find(".//pc:PC-Urn_label", ns)
                    if urn is None:
                        urn = prop.find(".//PC-Urn_label")
                    
                    # Get property value
                    value = prop.find(".//pc:PC-InfoData_value_sval", ns)
                    if value is None:
                        value = prop.find(".//PC-InfoData_value_sval")
                    if value is None:
                        value = prop.find(".//pc:PC-InfoData_value_ival", ns)
                    if value is None:
                        value = prop.find(".//PC-InfoData_value_ival")
                    if value is None:
                        value = prop.find(".//pc:PC-InfoData_value_fval", ns)
                    if value is None:
                        value = prop.find(".//PC-InfoData_value_fval")
                    
                    if urn is not None and value is not None and urn.text:
                        props[urn.text] = value.text
        
        if props:
            logging.info(f"Extracted {len(props)} properties from XML for CID: {cid}")
            return props
        else:
            logging.warning(f"No properties found in XML for CID: {cid}")
            return {}
    except Exception as e:
        logging.error(f"PubChem XML property lookup failed for CID: {cid} | Error: {e}")
        return {}
def _calculate_embedding_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    emb_a_norm = torch.nn.functional.normalize(emb_a, p=2, dim=-1)
    emb_b_norm = torch.nn.functional.normalize(emb_b, p=2, dim=-1)
    similarity = torch.cosine_similarity(emb_a_norm, emb_b_norm, dim=-1)
    return round(similarity.item(), 4)
def _get_neighbors(node_idx, edge_index):
    src, dst = edge_index
    neighbors = set(dst[src == node_idx].tolist())
    neighbors.update(src[dst == node_idx].tolist())
    return list(neighbors)
def _safe_smiles(val):
    if not isinstance(val, str):
        return ""
    if val.lower() == "nan":
        return ""
    return val
def predict(drug_a: str, drug_b: str, include_properties: bool = True) -> dict:
    def resolve_drug_id(drug_input: str) -> str:
        if drug_input in smiles_dict or drug_input in drug_to_node:
            return drug_input
        for drug_id, drug_name in drugid_to_name.items():
            if drug_name.lower() == drug_input.lower():
                return drug_id
        raise ValueError(f"Unknown drug: {drug_input}")
    try:
        drug_a = resolve_drug_id(drug_a)
        drug_b = resolve_drug_id(drug_b)
    except ValueError as e:
        raise e
    if drug_a not in smiles_dict and drug_a not in drug_to_node:
        raise ValueError(f"Unknown drug: {drug_a}")
    if drug_b not in smiles_dict and drug_b not in drug_to_node:
        raise ValueError(f"Unknown drug: {drug_b}")
    if drug_a not in drug_to_node or drug_b not in drug_to_node:
        smiles_a = _safe_smiles(smiles_dict.get(drug_a, ""))
        smiles_b = _safe_smiles(smiles_dict.get(drug_b, ""))
        # RDKit removed: return dummy similarity and probability
        similarity = 0.0
        probability = 0.0
        props_a = _extract_molecular_properties(smiles_a) if include_properties else {}
        props_b = _extract_molecular_properties(smiles_b) if include_properties else {}
        return {
            "probability": max(0.0, min(1.0, probability)),
            "embedding_similarity": similarity,
            "drug_a_properties": props_a,
            "drug_b_properties": props_b,
            "drug_a_smiles": smiles_a,
            "drug_b_smiles": smiles_b
        }
    # Removed OGB dataset loading and edge_index logic; only using precomputed artifacts
    i = drug_to_node[drug_a]
    j = drug_to_node[drug_b]
    z_i = node_embeddings[i].unsqueeze(0)
    z_j = node_embeddings[j].unsqueeze(0)
    z = torch.cat([z_i, z_j], dim=0)
    edge_pred = torch.tensor([[0], [1]])
    with torch.no_grad():
        logit_ab = decoder(z, edge_pred)
        logit_ba = decoder(z.flip(0), edge_pred)
        logit = (logit_ab + logit_ba) / 2
    probability = torch.sigmoid(logit).item()
    emb_similarity = _calculate_embedding_similarity(z_i, z_j)
    smiles_a = _safe_smiles(smiles_dict.get(drug_a, ""))
    smiles_b = _safe_smiles(smiles_dict.get(drug_b, ""))
    props_a = _extract_molecular_properties(smiles_a) if include_properties else {}
    props_b = _extract_molecular_properties(smiles_b) if include_properties else {}
    return {
        "probability": round(probability, 4),
        "embedding_similarity": emb_similarity,
        "drug_a_properties": props_a,
        "drug_b_properties": props_b,
        "drug_a_smiles": smiles_a,
        "drug_b_smiles": smiles_b
    }
