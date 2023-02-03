from model import Net
from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from WeightSchedule import WeighSchedule
from predictor import Predictor
from parameters import EPOCHS, BATCH_SIZE
from best_model_parameters import BestModelParameters
import torchvision
from Combined import CombinedModel

#EPOCHS = 1_000
#BATCH_SIZE = 512

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

teacher = Net().to(device)
student = Net().to(device)
#student.load_state_dict(teacher.state_dict())
predictor = Predictor(output=128, output_relu=False)
predictor_output = Predictor(output=128, output_relu=False)
teacher_predictor_mlp = Predictor(output=128, output_relu=False)


student_predictor = CombinedModel([
    student,
    predictor,
    predictor_output
]).to(device)

teacher_predictor = CombinedModel([
    teacher,
    teacher_predictor_mlp,
]).to(device)


optimizer = torch.optim.Adam(
    student.parameters(),
    lr=0.1
)

def learning_rate(epoch):
    return 3e-4
    if epoch < 10:
        return 0.3
    elif epoch < 20:
        return 0.2
    elif epoch < 30:
        return 0.1
    elif epoch < 40:
        return 0.01
    elif epoch < 50:
        return 0.001
    else:
        return 3e-4
    

#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#lambda1 = lambda epoch: learning_rate(epoch) #0.65 ** (epoch/10)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate)

weigh_schedule = WeighSchedule(
    student_predictor,
    teacher_predictor,
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
        p=1,
        ratio=(1, 1),
        scale=(0.03, 0.03)
    ),
    transforms.RandomHorizontalFlip(p=0.33),
    transforms.RandomVerticalFlip(p=0.33),
    transforms.RandomGrayscale(p=0.33),
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
    norm = lambda x: torch.linalg.norm(x, axis=-1)
    norm_x, norm_y = norm(x), norm(y)
    return -2. * torch.mean(torch.sum(x * y, axis=-1) / (norm_x * norm_y))

for epoch in range(EPOCHS):
    total_loss = torch.tensor(0.0, device=device)
    for (X, _) in dataloader:
        X = transform(X.to(device))
        X_masked = transform(X.clone())

        value = loss(student_predictor(X), teacher_no_grad(X_masked))
        value += loss(student_predictor(X_masked), teacher_no_grad(X))

        optimizer.zero_grad()
        value.backward()
        optimizer.step()        

        total_loss += value.item()
        weigh_schedule.update(epoch)
    print(f"epoch: {epoch}, total_loss: {total_loss}")
    best_params.set_loss_param(
        total_loss,
        student.state_dict(),
    )
    scheduler.step()

torch.save(
    student.state_dict(),
    "student_model" 
)
