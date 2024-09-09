import librosa
import torch
import torch.nn as nn
 
 
def variant_window(use_wp):
    x1_wp = [wp_t[0] for wp_t in use_wp]
    x2_wp = [wp_t[1] for wp_t in use_wp]
    
    x1_window_sizes = []
    x2_window_sizes = []

    f1_value = x1_wp[0] # value
    next_f1 = 0
    
    while next_f1 != (len(x1_wp) - 1):
        next_f1_value = f1_value+10 # x1 fixed_window_size = 10 or 20?
        f1 = x1_wp.index(f1_value)

        if next_f1_value not in x1_wp:
            next_f1 = len(x1_wp) - 1 # if iterations come here, then stop!
            x1_window = x1_wp[-1] - f1_value
        else:
            x1_window = 10
            next_f1 = x1_wp.index(f1_value+10) # index

        x2_window = x2_wp[next_f1] - x2_wp[f1] # value - value
        
        x1_window_sizes.append(x1_window)
        x2_window_sizes.append(x2_window)
        
        f1_value = next_f1_value
        
    return x1_window_sizes, x2_window_sizes
            

def variant_average_pooling(embeddings, window_sizes):
    """
    inputs: 
        embeddings = torch.Size([1, trimed_length, emb_dim])
        window_sizes = list []
        
    outputs:
        final_output = torch.Size([1, instance_len, emb_dim])
    """
    # List to store pooled results
    pooled_outputs = []
    emb_dim = int(embeddings.shape[2])

    # Perform average pooling over the specified window sizes
    for idx in range(len(window_sizes)):
        # Compute the number of segments to pool
        if idx == 0:
            segments = embeddings[:, :window_sizes[idx], :]
        else:
            segments = embeddings[:, sum(window_sizes[:idx]):sum(window_sizes[:idx])+window_sizes[idx], :]
        
        if segments.shape[1] == 0:
            pooled_data = torch.zeros(1, emb_dim)
        else:
            # Average Pooling, (1, seq_len, emb_dim) -> (1, 1, emb_dim)
            pool = nn.AdaptiveAvgPool2d((1, emb_dim))
            # Apply the pooling operation
            pooled_data = pool(segments).squeeze(0) # (1, emb_dim)
              
        # Store the result
        pooled_outputs.append(pooled_data)
            
        length = len(pooled_outputs)
        b = torch.Tensor(length, emb_dim) # (instance_len, emb_dim)
        new_pooled_outputs = torch.cat(pooled_outputs, out=b).unsqueeze(0)  
        
    return new_pooled_outputs # torch.Size([1,instance_len,emb_dim])
        
        
# character        
def window_mapping_and_character(x1_emb, x2_emb, x1_segments, x2_segments):
    """
    inputs: 
        x1_emb = torch.Size([1,seq_len,emb_dim]) = xx sec
        x2_emb = torch.Size([1,seq_len,emb_dim])
    outputs:
        mapped_x1_emb = torch.Size([1,instance_len,emb_dim])
        mapped_x2_emb = torch.Size([1,instance_len,emb_dim])
    """
    # 20ms = 1frame
    x1_segments = [[int(seg[0]*50), int(seg[1]*50)] for seg in x1_segments] 
    x2_segments = [[int(seg[0]*50), int(seg[1]*50)] for seg in x2_segments]

    # Define the pooling window sizes
    x1_window_sizes = [int(seg[1] - seg[0]) for seg in x1_segments]
    x2_window_sizes = [int(seg[1] - seg[0]) for seg in x2_segments]
    
    mapped_x1_emb = variant_average_pooling(x1_emb, x1_window_sizes)
    mapped_x2_emb = variant_average_pooling(x2_emb, x2_window_sizes)
    
    return mapped_x1_emb, mapped_x2_emb