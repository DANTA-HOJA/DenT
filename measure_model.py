import time

import torch
from fvcore.nn import FlopCountAnalysis

""" self-defined modules """
import DenT
from misc_utils import get_args
# -----------------------------------------------------------------------------/

# -------------------------
# Prepare arguments, variables
# -------------------------
device = "cuda"
args = get_args("train") # only for taking args
torch.cuda.empty_cache()

# -------------------------
# Preparing model
# -------------------------
print('===> Preparing model ...')

if args.model == 'DenT':
    model = DenT.DenseTransformer(args)
elif args.model == 'CusDenT':
    model = DenT.CustomizableDenT(
        add_pos_emb=args.add_pos_emb,
        use_multiheads=[bool(m) for m in args.use_multiheads]
    )
else:
    raise NotImplementedError

model = model.to(device)
model.eval()

# -------------------------
# IMPORTANT: use inference patch size
# -------------------------
dummy_input = torch.randn(1, 1, 32, 256, 256).to(device)

# -------------------------
# Params
# -------------------------
params = sum(p.numel() for p in model.parameters()) / 1e6

# -------------------------
# FLOPs (only once)
# -------------------------
flops = FlopCountAnalysis(model, dummy_input)
gflops = flops.total() / 1e9

# -------------------------
# Inference time
# -------------------------
with torch.no_grad():

    # warmup
    for _ in range(20):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    runs = 200
    start = time.time()
    for _ in range(runs):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()

latency = (end - start) / runs * 1000

# -------------------------
# Print
# -------------------------
print("========== Efficiency ==========")
print(f"Params: {params:.2f} M")
print(f"FLOPs: {gflops:.2f} GFLOPs")
print(f"Inference time: {latency:.2f} ms (batch=1)")
print("================================")