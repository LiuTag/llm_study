import _paths  # noqa: F401
import pytest
import torch

from Embedding import Embedding


def test_embedding_returns_selected_rows_with_arbitrary_leading_shape():
    layer = Embedding(num_embeddings=6, embedding_dim=3)
    weight = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    with torch.no_grad():
        layer.weight.copy_(weight)

    token_ids = torch.tensor([[0, 2], [5, 1]])
    output = layer(token_ids)

    assert output.shape == (2, 2, 3)
    assert torch.equal(output, weight[token_ids])


def test_embedding_rejects_negative_indices():
    layer = Embedding(num_embeddings=6, embedding_dim=3)

    with pytest.raises(ValueError):
        layer(torch.tensor([0, -1, 2]))


def test_embedding_weight_receives_gradients():
    layer = Embedding(num_embeddings=6, embedding_dim=3)
    output = layer(torch.tensor([1, 1, 4])).sum()
    output.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.count_nonzero(layer.weight.grad[1]) > 0
    assert torch.count_nonzero(layer.weight.grad[4]) > 0
    assert torch.count_nonzero(layer.weight.grad[0]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
