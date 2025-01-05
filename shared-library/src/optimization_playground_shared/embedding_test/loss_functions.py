import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, epsilon=1e-6):
        super(SimpleContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, embeddings, y=None):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        positives = torch.diagonal(similarity_matrix)

        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        negatives = similarity_matrix.masked_fill(mask, float('-inf'))
        negatives = torch.logsumexp(negatives, dim=1)

        # Todo: figure out why things turn negative
        loss = (negatives -positives).abs()

        return loss.mean()

class NegativeSample(torch.nn.Module):
    def __init__(self, temperature=0.1, epsilon=1e-6):
        super(NegativeSample, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, doc_embeddings, y=None):
        num_docs = doc_embeddings.shape[0]
        dot_product_matrix = torch.matmul(doc_embeddings, doc_embeddings.T)
        positive_scores = torch.diagonal(dot_product_matrix)
        negative_scores = dot_product_matrix - torch.eye(num_docs, device=doc_embeddings.device) * dot_product_matrix
        positive_loss = F.logsigmoid(positive_scores)
        negative_loss = F.logsigmoid(-negative_scores)
        loss = - (positive_loss + negative_loss.sum(dim=1)).mean()
    
        return loss

class MinimalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, epsilon=1e-6):
        super(MinimalCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, logits, y=None):
        labels = torch.arange(logits.shape[0])
        return torch.nn.functional.cross_entropy(logits, labels)

class NextTokenPrediction(torch.nn.Module):
    def __init__(self, padding_index):
        super(NextTokenPrediction, self).__init__()
        self.padding_index = padding_index

    def forward(self, logits, y):
        return F.cross_entropy(
            logits,
            y.reshape((-1)),
            ignore_index=self.padding_index,
        )

class TfIdfAnchor(torch.nn.Module):
    def __init__(self, documents):
        super(TfIdfAnchor, self).__init__()
        self.tf_idf = TfidfVectorizer(
            max_features=5_00,
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
            decode_error='replace', 
            strip_accents='unicode',
            norm='l2', 
            use_idf=True, 
            smooth_idf=True, 
            sublinear_tf=True,
        ).fit(documents)
    
    def forward(self, logits, documents):
        cosine_similarly_tf_idf = self.tf_idf.transform(documents)
        tfidf_dense = torch.tensor(cosine_similarly_tf_idf.toarray(), dtype=torch.float32).to(logits.device)

        norm_embeddings = F.normalize(logits, p=2, dim=-1).mean(dim=1)
        norm_tfidf = F.normalize(tfidf_dense, p=2, dim=-1).mean(dim=1)

        relative_differences_a = (norm_embeddings[1:] - norm_embeddings[:-1]) / norm_embeddings[:-1]
        relative_differences_b = (norm_tfidf[1:] - norm_tfidf[:-1]) / norm_tfidf[:-1]

        assert not torch.any(torch.isnan(relative_differences_a))
        assert not torch.any(torch.isnan(relative_differences_b))
        output = ((relative_differences_b - relative_differences_a) ** 2)
        assert not torch.any(torch.isnan(output))
        assert output.shape[0] > 0
        assert not torch.any(torch.isnan(output.mean()))

        return output.mean()
