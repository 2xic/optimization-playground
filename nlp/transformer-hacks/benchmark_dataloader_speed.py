from utils.web_dataloader import WebDataloader
import os
from dotenv import load_dotenv
import time

load_dotenv()


dataloader = WebDataloader(os.environ["WEB_DATALOADER"], "small-web", batch_size=1024)

start = time.time()
for index, (X, y) in enumerate(dataloader.iter()):
    if index > 100:
        break
print(time.time() - start)
