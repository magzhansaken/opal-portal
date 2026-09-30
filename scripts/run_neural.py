"""Train a neural model (OPAL, ablations, GRU, Transformer) for several seeds and save test predictions."""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from opal import data, train
from opal.metrics import fit_temperature

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="oulad"); p.add_argument("--kind", default="opal")
p.add_argument("--name", default=None); p.add_argument("--seeds", default="0,1,2,3,4")
p.add_argument("--cfg", default="{}"); p.add_argument("--threads", type=int, default=2)
a = p.parse_args()
torch.set_num_threads(a.threads)
cfg = json.loads(a.cfg)
name = a.name or a.kind
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", a.dataset, name)
os.makedirs(out_dir, exist_ok=True)
cs = data.load(a.dataset)
if a.dataset == "act-mooc":
    cfg.setdefault("steps_per_cohort", 8); cfg.setdefault("rec_steps", 8); cfg.setdefault("rec_learners", 1024)
for seed in [int(s) for s in a.seeds.split(",")]:
    f = os.path.join(out_dir, f"seed{seed}.pkl")
    if os.path.exists(f):
        continue
    t0 = time.time()
    ck = os.path.join(out_dir, f"seed{seed}.ckpt")
    model, hist = train.train(a.kind, cs, cfg, seed=seed, verbose=True, ckpt=ck)
    train_sec = float(sum(h.get("sec", 0.0) for h in hist))   # summed over epochs, robust to resumption
    rec = a.kind == "opal" and cfg.get("use_rec", True)
    val = train.evaluate(model, cs, split=1, rec=False)
    t1 = time.time()
    test = train.evaluate(model, cs, split=2, rec=rec)
    infer_sec = time.time() - t1
    temps = {h: fit_temperature(val[h]["logit"], val[h]["y"]) for h in val}
    n_params = sum(p.numel() for p in model.parameters())
    pickle.dump({"val": val, "test": test, "temps": temps, "hist": hist, "cfg": cfg, "train_sec": train_sec,
                 "infer_sec": infer_sec, "n_params": n_params}, open(f, "wb"))
    torch.save(model.state_dict(), os.path.join(out_dir, f"seed{seed}.pt"))
    if os.path.exists(ck):
        os.remove(ck)
    print("saved", f, round(train_sec, 1), "s", n_params, "params", flush=True)
