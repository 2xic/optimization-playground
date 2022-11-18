import torch.nn.functional as F
"""
loss

-> Predicted confidence map vs expected
-> Predicted parity fields vs expected

"""

def loss(annotation, predicted_confidence, predicted_paf, confidence, paf):
    # c -> Limbs 
    # p = 1
    # VERY naive implementation
    # print((predicted_confidence.shape, confidence.shape))
    # print((predicted_paf.shape, paf.shape))

    overfit = False

    predicted_confidence = F.interpolate(predicted_confidence, size=(480, 640), mode='bilinear')
    predicted_paf = F.interpolate(predicted_paf, size=(480, 640), mode='bilinear')

    if overfit:
        index = 3
        confidence  = ((predicted_confidence[0][index] - confidence[0][index]) ** 2).sum()
        paf_loss = 0
        return paf_loss + confidence

    predicted_paf = predicted_paf
    predicted_confidence = predicted_confidence  

    confidence_loss = 0
    for i in range(predicted_confidence.shape[0]):
      confidence_loss += (((predicted_confidence[i] - confidence[i])) ** 2).sum()

    paf_loss = 0
    for i in range(predicted_paf.shape[0]):
      paf_loss  += F.mse_loss(predicted_paf[0][2 * i], paf[0, i, :, :, 0])
      paf_loss  += F.mse_loss(predicted_paf[0][2 * i + 1], paf[0, i, :, :, 1])
    return (confidence_loss + paf_loss) / predicted_confidence.shape[0]
