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
            

def variant_average_pooling(embeddings, window_sizes, except_calculation):
    """
    inputs: 
        embeddings = torch.Size([1, trimed_length, 512])
        window_sizes = list []
        
    outputs:
        final_output = torch.Size([1, instance_len, 512])
    """
    # List to store pooled results
    pooled_outputs = []

    # Perform average pooling over the specified window sizes
    for idx in range(len(window_sizes)):
        if idx not in except_calculation:
            # Compute the number of segments to pool
            if idx == 0:
                segments = embeddings[:, :window_sizes[idx], :]
            else:
                segments = embeddings[:, sum(window_sizes[:idx]):sum(window_sizes[:idx])+window_sizes[idx], :]
            
            if segments.shape[1] == 0:
                pooled_data = torch.zeros(1, 512)
            else:
                # Average Pooling, (1, seq_len, 512) -> (1, 1, 512)
                pool = nn.AdaptiveAvgPool2d((1, 512))
                # Apply the pooling operation
                pooled_data = pool(segments).squeeze(0) # (1, 512)
                
            # Store the result
            pooled_outputs.append(pooled_data)
            
        length = len(pooled_outputs)
        b = torch.Tensor(length, 512) # (instance_len, 512)
        if len(pooled_outputs) != 0: # 이 case 왜 생기는지 모르겠음.
            new_pooled_outputs = torch.cat(pooled_outputs, out=b).unsqueeze(0)  
        else:
            new_pooled_outputs = None
    return new_pooled_outputs # torch.Size([1,instance_len,512])
        
# dtw        
def window_mapping_and_dtw(x1_emb, x2_emb, wp, x1_syllable, x2_syllable):
    """
    inputs: 
        x1_emb = torch.Size([1,1500,512]) = 30sec
        x2_emb = torch.Size([1,1500,512])
    outputs:
        mapped_x1_emb = torch.Size([1,instance_len,512])
        mapped_x2_emb = torch.Size([1,instance_len,512])
    """
    zeros = torch.zeros(1, 1, 512)
    
    if len(x1_syllable.shape) == 1:
        x1_start, x1_end = int(x1_syllable[0]), int(x1_syllable[1])
    else:
        x1_start, x1_end = int(x1_syllable[0][0]), int(x1_syllable[-1][1])

    if x1_start > 1500:
        return zeros, zeros
    elif x1_end > 1500:
        x1_end = 1499
        
    # x1: start - end
    wp_1500 = [[wp_t[0]//2, wp_t[1]//2] for wp_t in wp]
    x1_idx = [i[0] for i in wp_1500]

    START = x1_idx.index(x1_start)  # start frame_t
    END = x1_idx.index(x1_end) # end frame_t
    use_wp = wp_1500[START:END+1]
    
    # trim embeddings
    x1_emb = x1_emb[:, x1_start:x1_end+1, :] # [1, frames, dim]
    x2_emb = x2_emb[:, use_wp[0][1]:use_wp[-1][1], :]

    # Define the pooling window sizes
    x1_window_sizes, x2_window_sizes = variant_window(use_wp)
    
    # What if paired x2 window_size = 0?
    except_calculation_x1 = [i for i, value in enumerate(x1_window_sizes) if value == 0]
    except_calculation_x2 = [i for i, value in enumerate(x2_window_sizes) if value == 0]
    except_calculation = list(set().union(except_calculation_x1,except_calculation_x2))
    
    if len(except_calculation) == len(x1_window_sizes):
        return zeros, zeros
    else:
        # print(f"x1_window_sizes: {x1_window_sizes}")
        # print(f"x2_window_sizes: {x2_window_sizes}")
        # print(f"except_calculation: {except_calculation}")
        mapped_x1_emb = variant_average_pooling(x1_emb, x1_window_sizes, except_calculation)
        mapped_x2_emb = variant_average_pooling(x2_emb, x2_window_sizes, except_calculation)
        if (mapped_x1_emb != None) and (mapped_x2_emb != None):
            return mapped_x1_emb, mapped_x2_emb
        else:
            return zeros, zeros
        
# syllable
def window_mapping_and_syllable(x1_emb, x2_emb, x1_syllable, x2_syllable):
    """
    inputs: 
        x1_emb = torch.Size([1,1500,512]) = 30sec
        x2_emb = torch.Size([1,1500,512])
    outputs:
        mapped_x1_emb = torch.Size([1,instance_len,512])
        mapped_x2_emb = torch.Size([1,instance_len,512])
    """
    if len(x1_syllable.shape) == 1:
        # trim embeddings
        x1_emb = x1_emb[:, int(x1_syllable[0]):int(x1_syllable[1]), :] # [1, frames, dim]
        x2_emb = x2_emb[:, int(x2_syllable[0]):int(x2_syllable[1]), :]
        
        # Define the pooling window sizes
        x1_window_sizes = [int(x1_syllable[1])- int(x1_syllable[0])]
        x2_window_sizes = [int(x2_syllable[1]) - int(x2_syllable[0])]
    else:
        x1_emb = x1_emb[:, int(x1_syllable[0][0]):int(x1_syllable[-1][1]), :]
        x2_emb = x2_emb[:, int(x2_syllable[0][0]):int(x2_syllable[-1][1]), :]

        # Define the pooling window sizes
        x1_window_sizes = [int(seg[1] - seg[0]) for seg in x1_syllable]
        x2_window_sizes = [int(seg[1] - seg[0]) for seg in x2_syllable]

    
    # What if paired x2 window_size = 0?
    except_calculation_x1 = [i for i, value in enumerate(x1_window_sizes) if value == 0]
    except_calculation_x2 = [i for i, value in enumerate(x2_window_sizes) if value == 0]
    except_calculation = list(set().union(except_calculation_x1,except_calculation_x2))
    
    if len(except_calculation) == len(x1_window_sizes):
        zeros = torch.zeros(1, 1, 512)
        return zeros, zeros
    else:
        # print(f"x1_window_sizes: {x1_window_sizes}")
        # print(f"x2_window_sizes: {x2_window_sizes}")
        # print(f"except_calculation: {except_calculation}")
        mapped_x1_emb = variant_average_pooling(x1_emb, x1_window_sizes, except_calculation)
        mapped_x2_emb = variant_average_pooling(x2_emb, x2_window_sizes, except_calculation)
        if (mapped_x1_emb != None) and (mapped_x2_emb != None):
            return mapped_x1_emb, mapped_x2_emb
        else:
            zeros = torch.zeros(1, 1, 512)
            return zeros, zeros
        
# character        
def window_mapping_and_character(x1_emb, x2_emb, x1_segments, x2_segments):
    """
    inputs: 
        x1_emb = torch.Size([1,1500,512]) = 30sec
        x2_emb = torch.Size([1,1500,512])
    outputs:
        mapped_x1_emb = torch.Size([1,instance_len,512])
        mapped_x2_emb = torch.Size([1,instance_len,512])
    """
    x1_emb = x1_emb[:, int(x1_segments[1][0]):int(x1_segments[-2][1]), :]
    x2_emb = x2_emb[:, int(x2_segments[1][0]):int(x2_segments[-2][1]), :]

    # Define the pooling window sizes
    x1_window_sizes = [int(seg[1] - seg[0]) for seg in x1_segments[1:-1]]
    x2_window_sizes = [int(seg[1] - seg[0]) for seg in x2_segments[1:-1]]

    
    # What if paired x2 window_size = 0?
    except_calculation_x1 = [i for i, value in enumerate(x1_window_sizes) if value == 0]
    except_calculation_x2 = [i for i, value in enumerate(x2_window_sizes) if value == 0]
    except_calculation = list(set().union(except_calculation_x1,except_calculation_x2))
    
    if len(except_calculation) == len(x1_window_sizes):
        zeros = torch.zeros(1, 1, 512)
        return zeros, zeros
    else:
        # print(f"x1_window_sizes: {x1_window_sizes}")
        # print(f"x2_window_sizes: {x2_window_sizes}")
        # print(f"except_calculation: {except_calculation}")
        mapped_x1_emb = variant_average_pooling(x1_emb, x1_window_sizes, except_calculation)
        mapped_x2_emb = variant_average_pooling(x2_emb, x2_window_sizes, except_calculation)
        if (mapped_x1_emb != None) and (mapped_x2_emb != None):
            return mapped_x1_emb, mapped_x2_emb
        else:
            zeros = torch.zeros(1, 1, 512)
            return zeros, zeros