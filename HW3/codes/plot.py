import matplotlib.pyplot as plt
import json
import numpy as np
plt.figure()
heads = [1,3,6,12,24,48,96,192,384]
for head in heads:
    dir = f"head_{head}"
    with open(dir+"/epoch_val_loss.json") as f:
        val_loss = json.load(f)
        val_loss = val_loss[:6]
        plt.plot(val_loss,label="head="+str(head))
plt.xlabel("epoch")
plt.ylabel("val loss")
plt.title("loss with different num_head")
plt.legend()
plt.savefig("head.png")


dir1 = "Tfmr_finetune_shuffle"
dir2 = "Tfmr_finetune"
with open(dir1+"/epoch_train_loss.json") as f:
    train_loss = json.load(f)
with open(dir1+"/epoch_val_loss.json") as f:
    val_loss = json.load(f)
with open(dir1+"/epoch_ppl.json") as f:
    ppl = json.load(f)
with open(dir1+"/losses_by_iter.json") as f:
    all_losses = json.load(f)
with open(dir2+"/epoch_train_loss.json") as f:
    train_loss2 = json.load(f)
with open(dir2+"/epoch_val_loss.json") as f:
    val_loss2 = json.load(f)
with open(dir2+"/epoch_ppl.json") as f:
    ppl = json.load(f)
with open(dir2+"/losses_by_iter.json") as f:
    all_losses2 = json.load(f)
losses = []
for l in all_losses:
    losses += l
losses2 = []
for l in all_losses2:
    losses2 += l
val_loss = np.array(val_loss)
# linear interpolation
x = np.arange(0, len(val_loss))
print(len(losses))
x_new = np.linspace(0, len(val_loss), len(losses))
val_loss_new = np.interp(x_new, x, val_loss)


val_loss2 = np.array(val_loss2)
# linear interpolation
x2 = np.arange(0, len(val_loss2))
print(len(losses2))
x_new2 = np.linspace(0, len(val_loss2), len(losses2))
val_loss_new = np.interp(x_new2, x2, val_loss2)

plt.figure()
plt.plot(val_loss,label="shuffled")
plt.plot(val_loss2,label = "original")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("val loss")
plt.legend()
plt.savefig("compare_val_loss")


for dir in ["Tfmr_scratch","Tfmr_scratch_shuffle","Tfmr_finetune","Tfmr_finetune_shuffle"]:
    with open(dir+"/epoch_train_loss.json") as f:
        train_loss = json.load(f)
    with open(dir+"/epoch_val_loss.json") as f:
        val_loss = json.load(f)
    with open(dir+"/epoch_ppl.json") as f:
        ppl = json.load(f)
    with open(dir+"/losses_by_iter.json") as f:
        all_losses = json.load(f)
    losses = []
    for l in all_losses:
        losses += l
    
    val_loss = np.array(val_loss)
    # linear interpolation
    x = np.arange(0, len(val_loss))
    print(len(losses))
    x_new = np.linspace(0, len(val_loss), len(losses))
    val_loss_new = np.interp(x_new, x, val_loss)
    plt.figure()
    plt.plot(losses,label="train loss")
    plt.plot(val_loss_new,label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.title(dir)
    plt.savefig(dir+"/losses.png")
    
