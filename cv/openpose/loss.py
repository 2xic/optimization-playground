import torch.nn.functional as F
"""
loss

-> Predicted confidence map vs expected
-> Predicted parity fields vs expected

"""

def loss(predicted_confidence, predicted_paf, confidence, paf):
    # c -> Limbs 
    # p = 1
    # VERY naive implementation
  #  print((predicted_confidence.shape, confidence.shape))
 #   print((predicted_paf.shape, paf.shape))
    confidence_loss  = F.mse_loss(predicted_confidence, confidence)
    paf_loss  = F.mse_loss(predicted_paf, paf)
    
    return confidence_loss + paf_loss
