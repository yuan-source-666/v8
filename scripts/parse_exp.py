"""解析 v8 训练日志 → 结构化数据（合并段1+段2）。"""
import re, json

def parse_train_log(path):
    steps, losses, ppls, lrs, drv = [], [], [], [], []
    val = {}
    pat = re.compile(r"step (\d+)/\d+ loss ([\d.]+) \(EMA [\d.]+\) ppl ([\d.]+) lr ([\d.e+-]+)")
    dpat = re.compile(r"D ([\d.]+) desire ([\d.]+) fear ([\d.]+) drv_loss ([\d.\-]+) r ([+\d.\-]+) E ([\d.]+) R ([\d.]+) C ([\d.]+) M ([\d.]+)")
    vpat = re.compile(r"val_loss ([\d.]+) ppl ([\d.]+)")
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1))); losses.append(float(m.group(2)))
            ppls.append(float(m.group(3))); lrs.append(float(m.group(4)))
            d = dpat.search(line)
            if d:
                drv.append({"D": float(d.group(1)), "desire": float(d.group(2)),
                            "fear": float(d.group(3)), "drv_loss": float(d.group(4)),
                            "reward": float(d.group(5)), "energy": float(d.group(6)),
                            "resources": float(d.group(7)), "consistency": float(d.group(8)),
                            "safety": float(d.group(9))})
            else:
                drv.append(None)
        v = vpat.search(line)
        if v:
            val[int(steps[-1]) if steps else 0] = (float(v.group(1)), float(v.group(2)))
    return {"steps": steps, "losses": losses, "ppls": ppls, "lrs": lrs,
            "drives": drv, "val": val, "final_val": val.get(steps[-1] if steps else 0, None)}

base = parse_train_log("exp_logs/base_train.log")
d1 = parse_train_log("exp_logs/drives_1.log")   # step 10-200
d2 = parse_train_log("exp_logs/drives_2.log")   # step 210-400（含最终 eval）

# 合并 drives 曲线（去重 step，段2 覆盖）
allsteps = {}
for s, l, p, lr, dv in zip(d1["steps"], d1["losses"], d1["ppls"], d1["lrs"], d1["drives"]):
    allsteps[s] = {"loss": l, "ppl": p, "lr": lr, "drv": dv}
for s, l, p, lr, dv in zip(d2["steps"], d2["losses"], d2["ppls"], d2["lrs"], d2["drives"]):
    allsteps[s] = {"loss": l, "ppl": p, "lr": lr, "drv": dv}
dsteps = sorted(allsteps)
# 最终 eval 取段2（step 400）
final_val = d2["final_val"] or d1["final_val"]

out = {
    "meta": {"config": "exp (7.67M params, d_model=128, 4 layers, MMA layout)",
             "data": "domain corpus, 8492 tokens, GPT-2 BPE, seq_len=128, batch=8, grad_accum=2",
             "hardware": "2-core CPU, bf16 autocast, oneDNN, 400 iters, lr 1e-3 cosine"},
    "baseline": {"final_val_loss": base["final_val"][0], "final_ppl": base["final_val"][1],
                 "curve": [{"step": s, "loss": l, "ppl": p} for s, l, p, lr in zip(base["steps"], base["losses"], base["ppls"], base["lrs"])]},
    "drives": {"final_val_loss": final_val[0], "final_ppl": final_val[1],
               "curve": [{"step": s, "loss": allsteps[s]["loss"], "ppl": allsteps[s]["ppl"],
                          "D": allsteps[s]["drv"]["D"] if allsteps[s]["drv"] else None,
                          "desire": allsteps[s]["drv"]["desire"] if allsteps[s]["drv"] else None,
                          "fear": allsteps[s]["drv"]["fear"] if allsteps[s]["drv"] else None,
                          "reward": allsteps[s]["drv"]["reward"] if allsteps[s]["drv"] else None,
                          "energy": allsteps[s]["drv"]["energy"] if allsteps[s]["drv"] else None,
                          "consistency": allsteps[s]["drv"]["consistency"] if allsteps[s]["drv"] else None}
                          for s in dsteps]},
}
json.dump(out, open("exp_logs/experiment_data.json", "w"), ensure_ascii=False, indent=1)
print("baseline: val", out["baseline"]["final_val_loss"], "ppl", out["baseline"]["final_ppl"], "| points", len(out["baseline"]["curve"]))
print("drives:   val", out["drives"]["final_val_loss"], "ppl", out["drives"]["final_ppl"], "| points", len(out["drives"]["curve"]))
print("sample drives point:", out["drives"]["curve"][-1])
