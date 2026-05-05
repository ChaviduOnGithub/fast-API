"""
Unit tests for model_loader and models.
Run with: pytest fast-API/tests/test_model.py -v
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. EdgeDecoder (models.py)
# ---------------------------------------------------------------------------
class TestEdgeDecoder:
    @pytest.fixture
    def decoder(self):
        from models import EdgeDecoder
        return EdgeDecoder(hidden_channels=64)

    def test_output_shape_single_edge(self, decoder):
        z = torch.randn(2, 64)
        edge_index = torch.tensor([[0], [1]])
        out = decoder(z, edge_index)
        # single edge is squeezed to a scalar tensor
        assert out.shape == torch.Size([])

    def test_output_shape_batch(self, decoder):
        n = 50
        z = torch.randn(n, 64)
        src = torch.randint(0, n, (100,))
        dst = torch.randint(0, n, (100,))
        edge_index = torch.stack([src, dst])
        out = decoder(z, edge_index)
        assert out.shape == torch.Size([100])

    def test_sigmoid_output_in_range(self, decoder):
        z = torch.randn(2, 64)
        edge_index = torch.tensor([[0], [1]])
        logit = decoder(z, edge_index)
        prob = torch.sigmoid(logit).item()
        assert 0.0 <= prob <= 1.0

    def test_no_nan_in_output(self, decoder):
        z = torch.randn(10, 64)
        edge_index = torch.tensor([[i for i in range(9)], [i + 1 for i in range(9)]])
        out = decoder(z, edge_index)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))


# ---------------------------------------------------------------------------
# 2. _safe_smiles
# ---------------------------------------------------------------------------
class TestSafeSmiles:
    def test_normal_smiles(self):
        from model_loader import _safe_smiles
        assert _safe_smiles("CCO") == "CCO"

    def test_nan_string_returns_empty(self):
        from model_loader import _safe_smiles
        assert _safe_smiles("nan") == ""
        assert _safe_smiles("NaN") == ""

    def test_none_returns_empty(self):
        from model_loader import _safe_smiles
        assert _safe_smiles(None) == ""

    def test_non_string_returns_empty(self):
        from model_loader import _safe_smiles
        assert _safe_smiles(12345) == ""


# ---------------------------------------------------------------------------
# 3. _calculate_embedding_similarity
# ---------------------------------------------------------------------------
class TestEmbeddingSimilarity:
    def test_identical_vectors(self):
        from model_loader import _calculate_embedding_similarity
        v = torch.randn(1, 128)
        sim = _calculate_embedding_similarity(v, v)
        assert abs(sim - 1.0) < 1e-3

    def test_orthogonal_vectors(self):
        from model_loader import _calculate_embedding_similarity
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        sim = _calculate_embedding_similarity(a, b)
        assert abs(sim - 0.0) < 1e-3

    def test_output_in_valid_range(self):
        from model_loader import _calculate_embedding_similarity
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)
        sim = _calculate_embedding_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# 4. predict()
# ---------------------------------------------------------------------------
DRUG_A = "DB00001"
DRUG_B = "DB00002"
UNKNOWN = "DB99999"


class TestPredict:
    def test_known_pair_returns_dict(self):
        from model_loader import predict
        result = predict(DRUG_A, DRUG_B, include_properties=False)
        assert isinstance(result, dict)

    def test_probability_in_range(self):
        from model_loader import predict
        result = predict(DRUG_A, DRUG_B, include_properties=False)
        assert 0.0 <= result["probability"] <= 1.0

    def test_embedding_similarity_in_range(self):
        from model_loader import predict
        result = predict(DRUG_A, DRUG_B, include_properties=False)
        assert -1.0 <= result["embedding_similarity"] <= 1.0

    def test_symmetry(self):
        from model_loader import predict
        ab = predict(DRUG_A, DRUG_B, include_properties=False)["probability"]
        ba = predict(DRUG_B, DRUG_A, include_properties=False)["probability"]
        assert abs(ab - ba) < 1e-4

    def test_unknown_drug_raises_value_error(self):
        from model_loader import predict
        with pytest.raises(ValueError):
            predict(UNKNOWN, DRUG_B, include_properties=False)

    def test_no_nan_in_output(self):
        import math
        from model_loader import predict
        result = predict(DRUG_A, DRUG_B, include_properties=False)
        assert not math.isnan(result["probability"])
        assert not math.isnan(result["embedding_similarity"])

    def test_resolve_by_name(self):
        from model_loader import predict, drugid_to_name
        name_a = drugid_to_name.get(DRUG_A)
        if name_a:
            result = predict(name_a, DRUG_B, include_properties=False)
            assert 0.0 <= result["probability"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Artifact loading
# ---------------------------------------------------------------------------
class TestArtifacts:
    def test_node_embeddings_loaded(self):
        from model_loader import node_embeddings
        assert node_embeddings is not None
        assert node_embeddings.shape[0] > 0

    def test_decoder_loaded(self):
        from model_loader import decoder
        assert decoder is not None

    def test_drug_to_node_populated(self):
        from model_loader import drug_to_node
        assert len(drug_to_node) > 0

    def test_smiles_dict_populated(self):
        from model_loader import smiles_dict
        assert len(smiles_dict) > 0


# ---------------------------------------------------------------------------
# 6. risk_label (inference_api.py)
# ---------------------------------------------------------------------------
class TestRiskLabel:
    def test_low(self):
        from inference_api import risk_label
        assert risk_label(0.0) == "Low"
        assert risk_label(0.29) == "Low"

    def test_medium(self):
        from inference_api import risk_label
        assert risk_label(0.3) == "Medium"
        assert risk_label(0.59) == "Medium"

    def test_high(self):
        from inference_api import risk_label
        assert risk_label(0.6) == "High"
        assert risk_label(1.0) == "High"
