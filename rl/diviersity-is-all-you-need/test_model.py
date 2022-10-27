from diayn import *

model = DIAY(2, 2, 2)
model.sac.load_state_dict(
    torch.load('diayn_model')['model_state_dict']
)
print("Testing in env")
model.test_skills(SimpleEnv())
