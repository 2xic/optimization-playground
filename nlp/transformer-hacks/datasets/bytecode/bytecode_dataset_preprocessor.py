from .evm import get_opcodes_from_bytecode
import aiofiles
import asyncio
import json
import glob
import os


async def process_file(file_path):
    output_path = file_path.replace("contract_data", "evm-opcodes-data").replace(
        ".json", ".txt"
    )
    if os.path.isfile(output_path):
        return

    async with aiofiles.open(file_path, "r") as f:
        content = json.loads(await f.read())

    opcodes = get_opcodes_from_bytecode(
        bytes.fromhex(content["bytecode"].replace("0x", ""))
    )
    opcodes_str = "\n".join(map(str, opcodes))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async with aiofiles.open(output_path, "w") as f:
        await f.write(opcodes_str)


async def create_dataset_async(batch_size=50):
    files = glob.glob(
        "/home/brage/bigdrive/evm-contract-data/contract_data/**/*.json",
        recursive=True,
    )

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        await asyncio.gather(*[process_file(f) for f in batch], return_exceptions=True)
        print(
            f"Processed batch {i // batch_size + 1}/{(len(files) - 1) // batch_size + 1}"
        )


if __name__ == "__main__":
    asyncio.run(create_dataset_async())
