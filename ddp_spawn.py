import os
import sys
import torch
import torch.multiprocessing as mp

def _worker(rank: int, world_size: int, argv):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

 
    os.environ["GLOO_SOCKET_IFNAME"] = "Wi-Fi"

    # Import your train() entrypoint from train.py
    import train as train_module

    # Reuse the same CLI args parsing by calling train_module.train(args)
    # parses args the same way train.py does:
    args = train_module.parse_args()

    # Force cpu if requested in argv
    # (Your train.py already supports --cpu, so just pass it when running.)
    train_module.train(args)

def main():
    world_size = 2
    mp.spawn(_worker, args=(world_size, sys.argv[1:]), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
