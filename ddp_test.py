import os, torch, torch.distributed as dist, time
dist.init_process_group("nccl"); r = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(r); x = torch.full((1 << 26,), float(r + 1), device="cuda")
dist.all_reduce(x); t = time.time(); [dist.all_reduce(x) for _ in range(5)]; torch.cuda.synchronize(); print("rank", r, "ok GB/s", 5 * .268 / (time.time() - t), flush=True); dist.destroy_process_group()
