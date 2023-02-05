from components.model import Net
from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from components.WeightSchedule import WeighSchedule
from components.predictor import Predictor
from components.parameters import EPOCHS, BATCH_SIZE
from components.best_model_parameters import BestModelParameters
import torchvision
from components.combined import CombinedModel
import torch.nn.functional as F

def train(lock):
   # global EPOCHS
   # EPOCHS *= 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    teacher = Net().to(device)
    student = Net().to(device)
    output_shape = 8 * 8

    student_representation = Predictor(input=output_shape, output=output_shape, output_relu=False)
    student_prediction_of_teacher = Predictor(input=output_shape, output=output_shape, output_relu=False)
    teacher_representation = Predictor(input=output_shape, output=output_shape, output_relu=False)

    student_predictor = CombinedModel([
        student,
        student_representation,
        student_prediction_of_teacher
    ]).to(device)

    teacher_predictor = CombinedModel([
        teacher,
        teacher_representation,
    ]).to(device)

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=0.1,
        weight_decay=1.5*10e-6
    )

    def learning_rate(epoch):
        return 3e-4
        if epoch < 10:
            return 0.3
        elif epoch < 25:
            return 0.2
        elif epoch < 35:
            return 0.15
        elif epoch < 45:
            return 0.10
        elif epoch < 55:
            return 0.05
        elif epoch < 65:
            return 0.001
        else:
            return 3e-4

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate)

    weigh_schedule_encoder = WeighSchedule(
        student,
        teacher,
        EPOCHS
    )
    weigh_schedule_representation = WeighSchedule(
        student_representation,
        teacher_representation,
        EPOCHS
    )

    def teacher_no_grad(X):
        with torch.no_grad():
            return teacher_predictor(X)

    dataloader = DataLoader(
        Cifar10Dataloader(),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomErasing(
            p=0.3,
            ratio=(1, 1),
            scale=(0.03, 0.03)
        ),
        transforms.RandomHorizontalFlip(p=0.33),
        transforms.RandomVerticalFlip(p=0.33),
        transforms.RandomGrayscale(p=0.33),
        transforms.RandomApply([
            torchvision.transforms.GaussianBlur(9),
        ], p=0.3),
        transforms.RandomApply([
            torchvision.transforms.RandomRotation(180),
        ], p=0.3),
    ])

    grid_image = torchvision.utils.make_grid(
        transform(next(iter(dataloader))[0][:8])
    )
    torchvision.utils.save_image(
        grid_image,
        f"logs/example_transformation.png"
    )

    best_params = BestModelParameters()

    def loss(x, y):
        norm = lambda x: F.normalize(x, dim=1)
        norm_x, norm_y = norm(x), norm(y)
#        return torch.nn.MSELoss()(norm_x, norm_y)
        norm_sum = torch.sum(norm_x * norm_y, dim=-1)
        return torch.mean(2 - 2. * norm_sum)

    with open("logs/byol_loss.txt", "w") as file:
        for epoch in range(EPOCHS):
            total_loss = torch.tensor(0.0, device=device)
            for (X, _) in dataloader:
                X = (X.to(device))
                X_masked = transform(X.clone())

                value = loss(student_predictor(X), teacher_no_grad(X_masked))
                value += loss(teacher_no_grad(X), student_predictor(X_masked))

                optimizer.zero_grad()
                value.backward()
                optimizer.step()

                total_loss += value.item()
                weigh_schedule_representation.update(epoch)
                weigh_schedule_encoder.update(epoch)
            lock.acquire()
            try:
                print(f"BYOL - epoch: {epoch}, total_loss: {total_loss}")
            finally:
                lock.release()
            best_params.set_loss_param(
                total_loss,
                student.state_dict(),
            )
            file.write(f"{total_loss.item()}\n")
            scheduler.step()

            if epoch % 30 == 0 and epoch > 0:
                with torch.no_grad():
                    output = student(X)[:8].cpu().reshape((8, 1, 8, 8))

                    torchvision.utils.save_image(
                        output,
                        f"logs/example_student_encoder_epoch_{epoch}.png"
                    )

    torch.save(
        student.state_dict(),
      #  teacher.state_dict(),
        "student_model" 
    )

if __name__ == "__main__":
    from torch.multiprocessing import Lock
    lock = Lock()
    train(lock)
