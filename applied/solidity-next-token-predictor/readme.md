Train GPT like model on smart contract fiesta and see what happens.

## Setup
Run the ZMQ dataloader

```bash
python3 -m optimization_playground_shared.dataloaders.data_portal.Server --path "/mnt/blockstorage/smart-contract-fiesta/organized_contracts/**/**/*.sol"
```

## Need more scale
- Batch size
  - https://github.com/BlackHC/toma
  - https://pytorch-lightning.readthedocs.io/en/1.1.1/training_tricks.html#auto-scaling-of-batch-size

