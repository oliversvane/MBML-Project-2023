from torch import Tensor
import torch
import pandas as pd


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
        input_len = round(input_len) # just a precaution
        start_position = 0
        stop_position = num_obs-1 # because of 0 indexing
        
        subseq_first_idx = start_position
        subseq_last_idx = start_position + input_len
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len 
        print("target_last_idx is {}".format(target_last_idx))
        print("stop_position is {}".format(stop_position))
        indices = []
        while target_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size
            target_first_idx = subseq_last_idx + forecast_horizon
            target_last_idx = target_first_idx + target_len

        return indices

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        stop_position = len(data)-1 # 1- because of 0 indexing
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        
        subseq_last_idx = window_size
        
        indices = []
        
        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            
            subseq_last_idx += step_size

        return indices


def read_data(data: pd.DataFrame,  
    timestamp_col_name: str="timestamp") -> pd.DataFrame:
    data.set_index(timestamp_col_name)
    data = downcast_data(data)
    data.sort_values(by=[timestamp_col_name], inplace=True)
    return data




def downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    return df



import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple

class TransformerDataset(Dataset):
    def __init__(self, 
        data: torch.tensor,
        indices: list, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> None:

        super().__init__()
        self.indices = indices
        self.data = data
        print("From get_src_trg: data size = {}".format(data.size()))
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        
        return len(self.indices)

    def __getitem__(self, index):
        start_idx = self.indices[index][0]
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx]
        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y
    
    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
        src = sequence[:enc_seq_len] 
        trg = sequence[enc_seq_len:len(sequence)]
        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"
        trg_y = sequence[-target_seq_len:]
        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"
        return src, trg, trg_y.squeeze(-1)