import numpy as np
import torch
from skeleton import *
from model import *
import numpy as np

class PostProcess:
    def __init__(self) -> None:
        pass

    def process(self, predicted_confidence, predicted_paf, limbs=19):
        predicted_confidence = F.interpolate(
            predicted_confidence, size=(480, 640), mode='bilinear')
        predicted_paf = F.interpolate(
            predicted_paf, size=(480, 640), mode='bilinear')

        confidence_threshold = predicted_confidence[0].detach().numpy()
        for i in range(confidence_threshold.shape[0]):
            confidence_threshold[i] = self.confidence_threshold(confidence_threshold[i])
        confidence_threshold = torch.from_numpy(confidence_threshold)

        paf_merged = self.merge_paf(
            predicted_paf,
            limbs,
        )
        paf_threshold = paf_merged.detach().numpy()
        for i in range(limbs):
            paf_threshold[i] = self.paf_threshold(paf_merged[i])

        paf_threshold = torch.from_numpy(paf_threshold)
        
        return (
            confidence_threshold,
            paf_threshold
        )

    def confidence_threshold(self, x):
        x[np.abs(x) < 0.3] = 0
        return x

    def paf_threshold(self, x):
        x[np.abs(x) < 0.3] = 0
        return x
            
    def merge_paf(self, predicted_paf, limbs):
        merged = torch.zeros((limbs, 480, 640, 2))
        for i in range(limbs):
            merged[i, :, :, 0] = (torch.abs(predicted_paf[0, 2 * i, :, :]).detach())
            merged[i, :, :, 1] = (torch.abs(predicted_paf[0, 2 * i + 1, :, :]).detach())
        return merged
