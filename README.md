# polycoder_huggingface 

the idea here is to finally have hardware not be a major part of the process and design.
by implementing with huggingface all of those details can be just ignored (hopefuly) 

# issue: 
huggingface compute_metrics insists on loading all of the labels and logits of the ENTIRE dataset to memory before calculating the metrics 

# fixes:
1. use build in methods copute_metrics and preprocess_logits_for_metrics to precalculate the loss. this leave behind a weird artifact where all the labels are gathered in cpu memory and temporarly the losses in gpu memory. it should be fine since the labels are small enough and we can unload the logits 

2. overide the models forward method to return the sum on evaluation, then manualy calculate the preplexity afterwards
this should be the most computationaly optimal but its way too hacky.

# side notes:

# on 1:
I am not 100% sure that the acumelation works as advertised. it may still keep everything on cpu which will still crash things
# on 2:
note that the model avrages over the number of tokens in the batch...
evidance: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer_pt_utils.py#L473 line 509 to 512
so unfortunatly we cant use the simple method of just multiplying by the number of batches. 



TODO: add preplexity 