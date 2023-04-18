from optimization_playground_shared.training_loops.TrainingLoopCallback import TrainingLoopCallback
import torch

class StudentTeacherLoop(TrainingLoopCallback):
    def __init__(self, student, teacher, optimizer):
        self.teacher = teacher
        super().__init__(student, optimizer)

        self.kl_loss = torch.nn.CrossEntropyLoss() # torch.nn.KLDivLoss()
        self.nn_loss = torch.nn.CrossEntropyLoss() # torch.nn.NLLLoss()

    def loss(self, X, y_predicted, y_true):
        with torch.no_grad():
            predicted = self.teacher(X)
        teacher_loss = 0.5 * self.nn_loss(
            y_predicted,
            predicted,
        ) 
        class_loss = self.nn_loss(
            y_predicted,
            y_true
        )
        #return teacher_loss
        return (teacher_loss + class_loss) / 2
