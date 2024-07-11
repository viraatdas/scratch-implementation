import unittest
import torch

from model import MultiHeadAttentionBlock

class TestMultiHeadAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 32
        self.num_heads = 4
        self.dropout = 0.1

        self.B = 4  # batch size
        self.T = 8  # sequence length
        self.C = 32 # embedding dimension (d_model)

        self.mha_block = MultiHeadAttentionBlock(self.d_model, self.num_heads, self.dropout)

        self.q = torch.randn(self.B, self.T, self.C)
        self.k = torch.randn(self.B, self.T, self.C)
        self.v = torch.randn(self.B, self.T, self.C)

    def test_forward(self):
        output = self.mha_block(self.q, self.k, self.v, None)
        self.assertEqual(output.shape, (self.B, self.T, self.C))

if __name__ == '__main__':
    unittest.main()
