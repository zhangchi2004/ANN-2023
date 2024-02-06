import random
dir2 = "Tfmr_finetune_shuffle/"
dir1 = "Tfmr_scratch_shuffle/"
rn = random.sample(range(5000),10)

for dir in [dir1,dir2]:
    for ds in ["random_","top-p_"]:
        for tp in ["0.7","1.0"]:
            fname = dir+"output_"+ds+tp+".txt"
            with open(fname) as f:
                sentences = f.readlines()
            with open(dir+"output_"+ds+tp+"_sample.txt","w") as f:
                for i in rn:
                    f.write(sentences[i])
                