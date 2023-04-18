from .TrainingLoop import TrainingLoop
import torch
import torch.nn as nn

class TrainingLoopGenerated(TrainingLoop):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.loss = nn.NLLLoss()
        self.generated_loss = nn.MSELoss()

    def _iterate(self, dataloader, train=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model.to(device)
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)
        length = 0
        for row in dataloader:
            (X, y, generated) = (row[0], row[1], False)
            if train:
                generated = row[-1]

            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if train:
                loss = 0
                if 0 < y_pred[generated].shape[0]:
                    loss += 0.5 * self.generated_loss(nn.Softmax()(y_pred[generated]), nn.Softmax()(y[generated]))
                    #print("generated", y_pred[generated])
                    ##print("generated", y[generated])
                    #print("generated", loss)
                    #exit(0)
                    assert not torch.isnan(loss)
                if 0 < y_pred[generated == False].shape[0]:
                    loss += self.loss(y_pred[generated == False], torch.argmax(
                        y[generated == False], 1
                    ))
                    #print(loss)
                    assert not torch.isnan(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()      

                total_loss += loss  

            if train:
                accuracy += (torch.argmax(y_pred, 1) == torch.argmax(y, 1)).sum()
            else:
                accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100 
        return (
            total_loss,
            accuracy
        )
