import torch.nn.functional as F
"""
loss

-> Predicted confidence map vs expected
-> Predicted parity fields vs expected

"""


def debug_paf(index, predicted, paf):
    import matplotlib.pyplot as plt
    plt.clf()
    _, axarr = plt.subplots(1, 2, figsize=(15, 10))
    axarr[0].imshow(predicted.detach().numpy())
    axarr[1].imshow(paf)
    plt.savefig(f'debug/paf_{index}.png')


def debug_confidence(index, predicted, confidence):
    import matplotlib.pyplot as plt
    plt.clf()
    _, axarr = plt.subplots(1, 2, figsize=(15, 10))
    axarr[0].imshow(predicted.detach().numpy())
    axarr[1].imshow(confidence)
    plt.savefig(f'debug/confidence_{index}.png')


def loss(get_annotation, predicted_confidence, predicted_paf, confidence, paf):
    # c -> Limbs
    # p = 1
    # VERY naive implementation
    # print((predicted_confidence.shape, confidence.shape))
    # print((predicted_paf.shape, paf.shape))

    predicted_confidence = F.interpolate(
        predicted_confidence, size=(480, 640), mode='bilinear')
    predicted_paf = F.interpolate(
        predicted_paf, size=(480, 640), mode='bilinear')

    predicted_paf = predicted_paf
    predicted_confidence = predicted_confidence

    confidence_loss = 0
    for i in range(predicted_confidence.shape[1]):
        confidence_loss += (get_annotation(predicted_confidence.shape[1]) *
                            ((predicted_confidence[0][i] - confidence[0][i])) ** 2).sum()
        """
        debug_confidence(
            index=i,
            predicted=predicted_confidence[0][i],
            confidence=confidence[0][i]
        )
        """
    paf_loss = 0
    for i in range(predicted_paf.shape[1] // 2):
        """
        debug_paf(
            index=i,
            predicted=predicted_paf[0][2 * i],
            paf=paf[0, i, :, :, 0]
        )
        """
        paf_loss += F.mse_loss(get_annotation(1)
                               [0] * predicted_paf[0][2 * i], paf[0, i, :, :, 0])
        paf_loss += F.mse_loss(get_annotation(1)
                               [0] * predicted_paf[0][2 * i + 1], paf[0, i, :, :, 1])

    return (confidence_loss + paf_loss)
