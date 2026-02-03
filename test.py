import numpy as np, json, os

fold = "runs/prep/fold_P22"
Xtr = np.load(os.path.join(fold, "X_train.npy"))
ytr = np.load(os.path.join(fold, "y_train.npy"))
meta = json.load(open(os.path.join(fold, "meta.json")))

print(Xtr.shape, ytr.shape)
print("classes:", meta["class_list"])
print("y counts:", {i:int((ytr==i).sum()) for i in range(len(meta["class_list"]))})

# Check StandardScaler effect (training mean ~0, std ~1 per channel)
flat = Xtr.reshape(-1, Xtr.shape[-1])
print("mean abs (avg):", abs(flat.mean(axis=0)).mean())
print("std (avg):", flat.std(axis=0).mean())
