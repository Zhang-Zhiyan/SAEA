import torch
import torch.nn.functional as F

def top_filtering1(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
   
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
                                                                                           
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
                                                           
       
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                                                                       
        sorted_indices_to_remove = cumulative_probabilities > top_p
                                                                                         
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() 
        sorted_indices_to_remove[..., 0] = 0              
    
                                                            
        bt = logits.size(0)
        for i in range(bt):
            indices_to_remove =  sorted_indices[i][sorted_indices_to_remove[i]]             
            logits[i][indices_to_remove] = filter_value                       
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

