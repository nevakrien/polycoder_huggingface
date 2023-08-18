# polycoder_huggingface 

the idea here is to finally have hardware not be a major part of the process and design.
by implementing with huggingface all of those details can be just ignored (hopefuly) 

issue: huggingface compute_metrics insists on loading all of the labels and logits of the ENTIRE dataset to memory before calculating the metrics 
fixes:
1. overide the models forward method to return the sum on evaluation, then manualy calculate the preplexity afterwards
this should be the most computationaly optimal but its way too hacky.

2. use build in methods copute_metrics and preprocess_logits_for_metrics to precalculate the loss. this leave behind a weird artifact where all the labels are gathered in cpu memory and temporarly the losses in gpu memory. it should be fine since the labels are small enough and we can unload the logits 


note that the model avrages over the number of tokens in the batch...
evidance: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer_pt_utils.py#L473 line 509 to 512 

TODO: add preplexity 