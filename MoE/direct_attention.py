"""
STUB-A implementation: DirectedCrossAttention
 
Per-direction cross-attention modules for the A->C->B directed information flow.
Each direction gets its own Q/K/V projection matrices — no weight sharing.
 
Stop-gradient semantics (3.2):
  stop_grad_on_keys=True
      Applied on A->C and A->B paths.
      Group A keys/values are detached before K/V projection so Group A
      parameters receive NO gradient from cross-attention downstream paths.
      Group A must remain a stable sensory anchor — it cannot be shaped by
      the routing decisions of C or B.
 
  stop_grad_on_query_enrichment=True
      Applied on C->B path only.
      The cross-attention output (enriched B query) is detached from
      Group C's gradient graph before being returned to the caller.
      Prevents Group B's routing gradient from flowing back into Group C
      representations through this cross-attention path.
      (The C->B stop-grad in the router already blocks the gate path;
      this closes the second path through cross-attention.)
 
Integration in MoEBlock.forward (after shared self-attention, step 4.5):
    # Re-norm before cross-attention (separate LN pass, pre-normed inputs)
    tokens_A_ln2, tokens_B_ln2, tokens_C_ln2 = self.pre_exp_ln(
        tokens_A, tokens_B, tokens_C
    )
 
    tokens_C = tokens_C + self.cross_attn_A_to_C(tokens_C_ln2, tokens_A_ln2)
    tokens_B = tokens_B + self.cross_attn_A_to_B(tokens_B_ln2, tokens_A_ln2)
    tokens_B = tokens_B + self.cross_attn_C_to_B(tokens_B_ln2, tokens_C_ln2)
 
    # Note: tokens_C_ln2 passed to C->B is the PRE-RESIDUAL normed tensor,
    # not the post-residual tokens_C. This is intentional — B sees the
    # map context as it was before A's cross-attention updated it, which
    # keeps the gradient graph cleaner and prevents double-counting of
    # A->C information in the B path.
    # If you want B to see the A-enriched C representation, pass
    # self.pre_exp_ln_C(tokens_C) after applying the A->C residual.
    # This is left as a caller decision.
 
Required additions to MoEBlock.__init__:
    self.cross_attn_A_to_C = DirectedCrossAttention(
        embed_dim=D,
        num_heads=cfg.num_attn_heads,
        stop_grad_on_keys=True,
        stop_grad_on_query_enrichment=False,
        dropout=cfg.attn_dropout,
    )
    self.cross_attn_A_to_B = DirectedCrossAttention(
        embed_dim=D,
        num_heads=cfg.num_attn_heads,
        stop_grad_on_keys=True,
        stop_grad_on_query_enrichment=False,
        dropout=cfg.attn_dropout,
    )
    self.cross_attn_C_to_B = DirectedCrossAttention(
        embed_dim=D,
        num_heads=cfg.num_attn_heads,
        stop_grad_on_keys=False,
        stop_grad_on_query_enrichment=True,
        dropout=cfg.attn_dropout,
    )
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectedCrossAttention(nn.Module):
    """One directed cross-attention path (e.g. A -> C).
 
    Implements 3.2 steps 2-4.  Each direction (A->C, A->B, C->B) gets its own
    instance with independent Q/K/V projection matrices — NO weight sharing
    across directions.
 
    Pre-norm convention: the caller (MoEBlock) applies GroupLocalLayerNorm to
    both query_tokens and key_tokens before calling this module.  This module
    receives already-normed inputs and returns the cross-attention output ONLY
    (no residual).  The caller applies the residual:
        tokens_X = tokens_X + cross_attn(tokens_X_normed, tokens_Y_normed)
 
    Stop-gradient semantics:
        stop_grad_on_keys=True
            key_tokens are detached before Q/K/V projection.
            Use for A->C and A->B: Group A must not receive gradient from
            downstream cross-attention paths.
 
        stop_grad_on_query_enrichment=True
            The attention output is detached before being returned.
            Use for C->B: prevents Group B routing gradient from reaching
            Group C through this cross-attention path.
 
    Note on head count: num_heads may be fewer than the shared self-attention
    heads (the plan says "may be fewer").  A common choice is half.
 
    Args:
        embed_dim:                     D — feature dimension (Q, K, V all D)
        num_heads:                     H — attention heads
        stop_grad_on_keys:             detach key_tokens before K/V projection
        stop_grad_on_query_enrichment: detach output from key_tokens grad graph
        dropout:                       attention weight dropout
    """

    def __init__(
            self, 
            embed_dim: int,
            num_heads: int,
            stop_grad_on_keys: bool = False,
            stop_grad_on_query_enrichment: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim} must be divisible by num_heads ({num_heads}))"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.stop_grad_on_keys = stop_grad_on_keys
        self.stop_grad_on_query_enrichment = stop_grad_on_query_enrichment

        # independent per-direction projections (no weight sharing across instances)
        # no bias - consistent with GateExpertFFN
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)


        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()


    def _init_weights(self) -> None:
        """Xavier uniform for Q/K/V: small gain for out_proj (resiudal stream)"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        # small out_proj so cross-attention starts as a weak perturbation.
        # the residual stream dominiates early training, letting shared
        # self-attention stabilise before cross attention gradients grow.
        nn.init.xavier_uniform_(self.out_proj.weight, gain = 0.02)

    def forward(
            self,
            query_tokens: torch.Tensor,         #(B, N_query, D) - pre-normed
            key_tokens: torch.Tensor,           #(B, N_key,   D) - pre-normed
            key_padding_mask: Optional[torch.Tensor] = None,    # (B, N_key) bool, True = ignore
    )-> torch.Tensor:
        """
        Compute cross-attention output.
        Args:
            query_tokens:      (B, N_query, D) — already group-local normed
            key_tokens:        (B, N_key,   D) — already group-local normed
            key_padding_mask:  (B, N_key) bool, True = padding position (ignored)
 
        Returns:
            (B, N_query, D) — cross-attention output ONLY, NO residual.
            The caller adds this to the pre-residual query tokens:
                tokens_X = tokens_X + self.cross_attn_Y_to_X(normed_X, normed_Y)        
        """
        B, N_q, D = query_tokens.shape
        N_k = key_tokens.shape[1]

        # stop gradient on keys (A -> C, A -> B paths)
        # group A weights must not receive gradient from this path.

        keys_for_proj = key_tokens.detach() if self.stop_grad_on_keys else key_tokens

        # Q/K/V projections - each directions has its own weight matrices.
        Q = self.q_proj(query_tokens) # (B, N_q, D)
        K = self.k_proj(keys_for_proj) # (B, N_k, D)
        V = self.v_proj(keys_for_proj)  # (B, N_k, D)

        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, hd)
        K = K.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, hd)
        V = V.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, hd)

        # scaled dot-product attention.

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # (B, H, N_q, N_k)

        if key_padding_mask is not None:
            # key padding mask: (B, N_k) true = ignore
            # expand to (B, 1, 1, N_k) for broadcast over heads and queries
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N_k)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim = -1) # (B, H, N_q, N_k)
        attn_weights = self.attn_dropout(attn_weights)

        # weighted sum of values.

        attn_out = torch.matmul(attn_weights, V) # (B, H, N_q, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, N_q, D) # (B, N_q, D)

        # projection
        out = self.out_proj(attn_out) # (B, N_q, D)

        # Stop-gradient on query enrichment (C->B path).
        # Detach the output so Group B routing gradient cannot flow back
        # into Group C representations through this cross-attention path.
        if self.stop_grad_on_query_enrichment:
            out = out.detach()

        return out
    
    
if __name__ == "__main__":
    torch.manual_seed(42)
    B, D, H = 2, 128, 4
    N_A, N_B, N_C = 32, 16, 32
 
    # ── A->C: stop_grad_on_keys=True 
    # Fresh leaf tensors for each test to avoid accumulated .grad from prior runs.
    tA = torch.randn(B, N_A, D, requires_grad=True)
    tC = torch.randn(B, N_C, D, requires_grad=True)
 
    attn_A_to_C = DirectedCrossAttention(D, H, stop_grad_on_keys=True)
    out_AC = attn_A_to_C(tC, tA)
    assert out_AC.shape == (B, N_C, D), f"A->C shape: {out_AC.shape}"
 
    # Gradient should reach tC (query) but NOT tA (key was detached).
    out_AC.sum().backward()
    assert tC.grad is not None, "tC should have grad (it's the query)"
    assert tA.grad is None,     "tA should have NO grad (stop_grad_on_keys=True)"
    print(f"A->C  output shape : {out_AC.shape}  ✓")
    print(f"A->C  query grad   : {tC.grad is not None}  (expect True)   ✓")
    print(f"A->C  key   grad   : {tA.grad is None}       (expect True — detached) ✓")
 
    # ── A->B: stop_grad_on_keys=True
    tA2 = torch.randn(B, N_A, D, requires_grad=True)
    tB  = torch.randn(B, N_B, D, requires_grad=True)
 
    attn_A_to_B = DirectedCrossAttention(D, H, stop_grad_on_keys=True)
    out_AB = attn_A_to_B(tB, tA2)
    assert out_AB.shape == (B, N_B, D), f"A->B shape: {out_AB.shape}"
 
    out_AB.sum().backward()
    assert tB.grad  is not None, "tB should have grad (query)"
    assert tA2.grad is None,     "tA should have NO grad (detached key)"
    print(f"\nA->B  output shape : {out_AB.shape}  ✓")
    print(f"A->B  query grad   : {tB.grad is not None}  (expect True)   ✓")
    print(f"A->B  key   grad   : {tA2.grad is None}       (expect True — detached) ✓")
 
    # ── C->B: stop_grad_on_query_enrichment=True
    # The output is .detach()-ed, so it carries no grad_fn.
    # We verify this by checking requires_grad on the output and confirming
    # that wiring it into a downstream sum does not give grads to tB2/tC2.
    tB2 = torch.randn(B, N_B, D, requires_grad=True)
    tC2 = torch.randn(B, N_C, D, requires_grad=True)
 
    attn_C_to_B = DirectedCrossAttention(D, H, stop_grad_on_query_enrichment=True)
    out_CB = attn_C_to_B(tB2, tC2)
    assert out_CB.shape == (B, N_B, D), f"C->B shape: {out_CB.shape}"
 
    # Output must not require grad (it was detached).
    assert not out_CB.requires_grad, \
        "C->B output should not require_grad (stop_grad_on_query_enrichment)"
 
    # Wire into a residual sum and backward through it — no grad should reach
    # the original tB2/tC2 via the cross-attention path.
    residual_stream = tB2 + out_CB    # tB2 still has requires_grad=True
    residual_stream.sum().backward()
    # tB2.grad comes entirely from the identity path (d(tB2 + const)/d(tB2) = 1)
    # It should be all-ones, NOT propagated through the cross-attn computation.
    assert tB2.grad is not None, "tB2 should have grad via direct residual path"
    assert torch.allclose(tB2.grad, torch.ones_like(tB2)), \
        "tB2 grad should be all-ones (identity path only, not cross-attn path)"
    assert tC2.grad is None, \
        "tC2 should have NO grad (cross-attn output was detached)"
    print(f"\nC->B  output shape            : {out_CB.shape}  ✓")
    print(f"C->B  output.requires_grad    : {out_CB.requires_grad}  (expect False) ✓")
    print(f"C->B  tB2 residual grad=ones  : {torch.allclose(tB2.grad, torch.ones_like(tB2))}  ✓")
    print(f"C->B  tC2.grad is None        : {tC2.grad is None}  (expect True) ✓")
 
    # ── Key padding mask 
    tA3 = torch.randn(B, N_A, D)
    tC3 = torch.randn(B, N_C, D)
    pad_mask = torch.zeros(B, N_A, dtype=torch.bool)
    pad_mask[0, -4:] = True   # last 4 A-tokens in sample 0 are padding
    attn_with_mask = DirectedCrossAttention(D, H, stop_grad_on_keys=True)
    out_masked = attn_with_mask(tC3, tA3, key_padding_mask=pad_mask)
    assert out_masked.shape == (B, N_C, D)
    assert not torch.isnan(out_masked).any(), "NaN in masked output"
    print(f"\nKey padding mask : shape {out_masked.shape}, no NaN  ✓")
 
    # ── Independent weights — three directions must not share parameters ──
    attn1 = DirectedCrossAttention(D, H, stop_grad_on_keys=True)
    attn2 = DirectedCrossAttention(D, H, stop_grad_on_keys=True)
    attn3 = DirectedCrossAttention(D, H, stop_grad_on_query_enrichment=True)
    # Mutate attn1's q_proj; attn2 must be unaffected
    with torch.no_grad():
        attn1.q_proj.weight.fill_(999.0)
    assert not torch.allclose(attn1.q_proj.weight, attn2.q_proj.weight), \
        "FAIL: attn1 and attn2 are sharing parameters"
    print(f"\nWeight independence: attn1 ≠ attn2  ✓")
 
    # ── out_proj small-gain init sanity
    attn_check = DirectedCrossAttention(D, H)
    out_std = attn_check.out_proj.weight.std().item()
    q_std   = attn_check.q_proj.weight.std().item()
    assert out_std < q_std * 0.5, \
        f"out_proj should have smaller std than q_proj, got {out_std:.4f} vs {q_std:.4f}"
    print(f"\nout_proj init (std={out_std:.4f}) < q_proj init (std={q_std:.4f})  ✓")
 
    print("\n" + "="*55)
    print("All DirectedCrossAttention tests PASSED.")
    print("="*55)
 