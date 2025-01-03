import asyncio 
from tqdm import tqdm

async def gather_batch(items, operator, batch_size=5, label=""):
    batch = []
    index = 0
    with tqdm(total=len(items), desc=label) as pbar:
        while index < len(items):
            if len(batch) >= batch_size:
                results = await asyncio.gather(*batch)
                for v in results:
                    yield v
                batch = []
            batch.append(operator(items[index]))
            index += 1
            pbar.update(1)
        results = await asyncio.gather(*batch)
        for v in results:
            yield v
