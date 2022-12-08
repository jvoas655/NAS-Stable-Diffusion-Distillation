import json
import random
import os
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter

def mutate_encoding(base_encoding, num_mutates):
    channel_params = base_encoding["channel_masks"]
    encoding = base_encoding
    section = random.choice([".input_blocks.", ".middle_blocks.", ".out.", ".output_blocks.", ".time_embed."])
    block = random.randint(0,11)
    for key in channel_params:
        split_key = key.split(".")
        param_type = split_key[-1]
        param_base = ".".join(split_key[:-1])
        if (section not in key):
            continue
        if ("blocks" in section and f"blocks.{block}." not in key):
            continue
        param_info = channel_params[key]
        for dim in param_info:
            mask = param_info[dim]["mask"]
            for m in range(num_mutates * param_info[dim]["mult"]):
                new = -1
                while (new == -1 or new in mask):
                    new = random.randint(0, param_info[dim]["base"] * param_info[dim]["mult"] - 1)
                target = random.randint(0, len(mask) - 1)
                mask[target] = new

            encoding["channel_masks"][key][dim]["mask"] = mask
    return encoding

def enforce_wb_pairs(encoding):
    channel_params = base_encoding["channel_masks"]
    for key in channel_params:
        split_key = key.split(".")
        param_type = split_key[-1]
        param_base = ".".join(split_key[:-1])
        if (param_type == "bias"):
            #print(encoding["channel_masks"][key])
            encoding["channel_masks"][key]["0"]["mask"] = channel_params[param_base + ".weight"]["0"]["mask"]
    return encoding
if __name__ == "__main__":
    exp_name = "enf_wb_p_288"
    current_enc_path = "..\\logs\\" + exp_name + "\\current_encoding.json"
    channel_mask_size = 288
    num_mutates = 2
    sample_pop = 3
    num_epochs = 25000
    mult_method = "strided" # random, adjacent, strided
    init_method = "random" # random, first
    with open("..\\base_encoding.json", "r") as fileref:
        base_encoding = json.loads(fileref.read())
    params = []
    channel_params = base_encoding["channel_masks"]
    encoding = base_encoding
    encoding["num_masked_channels"] = channel_mask_size
    for key in channel_params:
        split_key = key.split(".")
        param_type = split_key[-1]
        param_base = ".".join(split_key[:-1])

        param_info = channel_params[key]
        for dim in param_info:
            mask = random.sample(range(param_info[dim]["base"] * param_info[dim]["mult"]), channel_mask_size * param_info[dim]["mult"])
            encoding["channel_masks"][key][dim]["mask"] = mask
    encoding = enforce_wb_pairs(encoding)
    if (os.path.exists("..\\logs\\" + exp_name)):
        shutil.rmtree("..\\logs\\" + exp_name)
    os.mkdir("..\\logs\\" + exp_name)
    writer = SummaryWriter("..\\logs\\" + exp_name, comment=exp_name)
    with open(current_enc_path, "w") as fileref:
        fileref.write(json.dumps(encoding))
    df = pd.DataFrame(columns=["ind", "VLB", "loss"])
    for epoch in range(num_epochs):
        
        os.chdir("..\\InvokeAI")
        
        os.system(f"python main.py --base .\\configs\\stable-diffusion\\v1-finetune.yaml --actual_resume .\\models\\ldm\\stable-diffusion-v1\\v1-5-pruned-emaonly.ckpt -n {exp_name}_search_{epoch} --gpus 0, --encoding {current_enc_path}")
        for root, dirs, files in os.walk("logs"):
            if (f"{exp_name}_search_{epoch}" in root):
                res = pd.read_csv(root + "\\csv\\version_0\\metrics.csv")
                df.loc[len(df.index)] = [epoch, res.at[0, "val/loss_vlb"], res.at[0, "val/loss"]]
                shutil.rmtree(root)
                break
        df.to_csv("..\\logs\\" + exp_name + "\\metrics.csv")
        
        writer.add_scalar(f"vlb", df.loc[df["VLB"].idxmax()].tolist()[0], epoch)
        
        with open("..\\logs\\" + exp_name + f"\\epoch_{epoch}_encoding.json", "w") as fileref:
            fileref.write(json.dumps(encoding))
        vlbs_inds = df["VLB"].sort_values(ascending=True).index.values
        if (len(vlbs_inds) > sample_pop):
            vlbs_inds = vlbs_inds[:sample_pop]
        with open("..\\logs\\" + exp_name + f"\\epoch_{random.choice(vlbs_inds)}_encoding.json", "r") as fileref:
            encoding = json.loads(fileref.read())
        
        os.chdir("..\\src")
        encoding = mutate_encoding(encoding, num_mutates)
        encoding = enforce_wb_pairs(encoding)
        with open(current_enc_path, "w") as fileref:
            fileref.write(json.dumps(encoding))
        

    



