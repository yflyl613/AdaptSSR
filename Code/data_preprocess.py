# %%
import os
import pickle
import random
from tqdm import tqdm

# %%
item_map = {0: 0}
with open("../Data/original_desen_pretrain.csv") as f:
    lines = f.readlines()

new_lines = []
for line in tqdm(lines):
    behavior = [int(x) for x in line.strip("\n").split(",")]
    for item in behavior:
        if item not in item_map:
            item_map[item] = len(item_map)
    behavior_remap = [str(item_map[item]) for item in behavior]
    new_lines.append(",".join(behavior_remap) + "\n")

print("n_items:", len(item_map))
with open("../Data/item_map.pkl", "wb") as f:
    pickle.dump(item_map, f, pickle.HIGHEST_PROTOCOL)

with open(f"../Data/pretrain_remap.csv", "w") as f:
    f.writelines(new_lines)

# %%
idx = [str(x) for x in range(len(lines))]
random.seed(2309)
random.shuffle(idx)
total_lines = len(lines)
train_lines = idx[: int(total_lines * 0.9)]
val_lines = idx[int(total_lines * 0.9) :]

with open("../Data/pretrain_train.csv", "w") as f:
    f.write("\n".join(train_lines) + "\n")
with open("../Data/pretrain_val.csv", "w") as f:
    f.write("\n".join(val_lines) + "\n")

with open("../Data/pretrain_train.csv") as f:
    lines = f.readlines()

os.makedirs("../Data/pretrain_train", exist_ok=True)
for i in range(20):
    random.seed(2241 + i)
    random.shuffle(lines)
    with open(f"../Data/pretrain_train/{i}.csv", "w") as f:
        f.writelines(lines)

# %%
with open("../Data/pretrain_remap.csv") as f:
    lines = f.readlines()

behavior_record = {}
for uid, line in enumerate(tqdm(lines)):
    behavior = line.strip("\n")
    behavior_record[behavior] = uid

# %%
for task in ["age", "lifestatus"]:
    with open(f"../Data/original_desen_{task}.csv") as f:
        lines = f.readlines()

    label_map = {}
    new_lines = []
    for line in tqdm(lines):
        behavior, label = line.strip("\n").split(",,")
        behavior_remap = ",".join(
            [str(item_map[item]) for item in [int(x) for x in behavior.split(",")]]
        )
        label = int(label)
        if label not in label_map:
            label_map[label] = len(label_map)
        label_remap = label_map[label]
        new_lines.append(
            str(behavior_record[behavior_remap]) + " " + str(label_remap) + "\n"
        )

    dirname = f"../Data/{task}"
    os.makedirs(dirname, exist_ok=True)

    with open(os.path.join(dirname, f"{task}_label_map.pkl"), "wb") as f:
        pickle.dump(label_map, f)
    with open(os.path.join(dirname, f"{task}_remap.csv"), "w") as f:
        f.writelines(new_lines)

    random.seed(1730)
    random.shuffle(new_lines)
    total_lines = len(new_lines)
    train_lines = new_lines[: int(total_lines * 0.6)]
    val_lines = new_lines[int(total_lines * 0.6) : int(total_lines * 0.8)]
    test_lines = new_lines[int(total_lines * 0.8) :]

    with open(os.path.join(dirname, "finetune_train.csv"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(dirname, "finetune_val.csv"), "w") as f:
        f.writelines(val_lines)
    with open(os.path.join(dirname, "finetune_test.csv"), "w") as f:
        f.writelines(test_lines)


# %%
for task in ["like", "click"]:
    with open(f"../Data/original_desen_finetune_{task}_nouserID.csv") as f:
        lines = f.readlines()

    label_map = {}
    new_lines = []
    for line in tqdm(lines):
        behavior, labels = line.strip("\n").split(",,")
        behavior_remap = ",".join(
            [str(item_map[item]) for item in [int(x) for x in behavior.split(",")]]
        )
        labels = [int(x) for x in labels.split(",")]
        for label in labels:
            if label not in label_map:
                label_map[label] = len(label_map)
        label_remap = [str(label_map[label]) for label in labels]
        uid = str(behavior_record[behavior_remap])
        for label in label_remap:
            new_lines.append(uid + " " + label + " " + ",".join(label_remap) + "\n")

    dirname = f"../Data/{task}"
    os.makedirs(dirname, exist_ok=True)

    with open(os.path.join(dirname, f"{task}_label_map.pkl"), "wb") as f:
        pickle.dump(label_map, f)
    with open(os.path.join(dirname, f"{task}_remap.csv"), "w") as f:
        f.writelines(new_lines)

    random.seed(1354)
    random.shuffle(new_lines)
    total_lines = len(new_lines)
    train_lines = new_lines[: int(total_lines * 0.6)]
    val_lines = new_lines[int(total_lines * 0.6) : int(total_lines * 0.8)]
    test_lines = new_lines[int(total_lines * 0.8) :]

    with open(os.path.join(dirname, "finetune_train.csv"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(dirname, "finetune_val.csv"), "w") as f:
        f.writelines(val_lines)
    with open(os.path.join(dirname, "finetune_test.csv"), "w") as f:
        f.writelines(test_lines)

# %%
