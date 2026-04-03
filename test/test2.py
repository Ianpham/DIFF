# import torch

# attention_maps = torch.randn([2, 7,5])
# B, K, N = attention_maps.shape
# winners = attention_maps.argmax(dim =1)
# print('winner shape', winners.shape)
# print('winner value', winners[0])

# quality_score = []

# for k in range(K):
#         winning_mask = (winners == k).float() # [B, N]
#         print('winning_mask', winning_mask)
#         # slot k's attention on all tokens
#         slot_k_attn = attention_maps[:,k,:] #[B, N]

#         # attention on winning tokens
#         winning_attn = (slot_k_attn * winning_mask).sum(dim = 1) # [B]
#         print('winning-attn', winning_attn)
#         total_attn = slot_k_attn.sum(dim = 1) # [B]

#         # quality = winning/total
#         quality = winning_attn/ (total_attn + 1e-8)
#         quality_score.append(quality)

#         final = torch.stack(quality_score, dim = 1) # [B, K]


import torch

B, K, N = 2, 3, 4  # 2 batches, 3 slots, 4 tokens
attention_maps = torch.randn(B, K, N)

print("attention_maps shape:", attention_maps.shape)  # [2, 3, 4]

# For one specific token (e.g., batch 0, token 0):
print("Attention scores for batch 0, token 0 across all slots:")
print(attention_maps[0, :, 0])  # 3 values (one per slot)

winners = attention_maps.argmax(dim=1)
print("\nwinners shape:", winners.shape)  # [2, 4]
print("Winner slot for batch 0, token 0:", winners[0, 0])