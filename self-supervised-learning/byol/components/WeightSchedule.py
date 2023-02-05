import torch

class WeighSchedule:
    def __init__(self, student, teacher, epochs):
        self.student = student
        self.teacher = teacher

        self.base = 0.996
        self.K = epochs
        self.t = self.update_t(0)

    def update_t(self, epoch):
        cosine = (torch.cos(
            torch.tensor(torch.pi * epoch / self.K)
        ) + 1) / 2
        self.t = (1 - (1 - self.base)) * cosine
        return self.t

    def update(self, epoch):
        self.update_t(epoch)
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.copy_(
                self.t*teacher_param.data + (1 - self.t) * student_param.data
            )
