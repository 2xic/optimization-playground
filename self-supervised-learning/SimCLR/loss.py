import torch
import torch.nn.functional as F

class Loss:
	def __init__(self) -> None:
		self.temperature = 0.5

	def loss(self, z_1, z_2):
		device = z_1.device
		"""
		Note so the problem is the cross comparing
		"""
		batch_size = z_1.shape[0]
		mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=device)).float()

		S = self.debug_calc_similarity_batch(z_1, z_2)

		sim_ij = torch.diag(S, batch_size).to(device)
		sim_ji = torch.diag(S, -batch_size).to(device)
		positives = torch.cat([sim_ij, sim_ji], dim=0).to(device)

		nominator = torch.exp(positives / self.temperature)
		denominator = mask * torch.exp(S / self.temperature)

		all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
		loss = torch.sum(all_losses) / (2 * batch_size)

		return loss

	def debug_calc_similarity_batch(self, a, b):
		"""
		Note: this is where the bug was in the training loop
		"""
		representations = torch.cat([a, b], dim=0)
		return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
