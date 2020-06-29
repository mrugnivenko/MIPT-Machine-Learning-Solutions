
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()

        self.title_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)

        self.conv1 = nn.Conv1d(in_channels=hid_size, out_channels=1, kernel_size=2)
        self.relu = nn.ReLU()
        self.adapt_avg_pool =  nn.AdaptiveAvgPool1d(output_size = concat_number_of_features//3)


        self.category = nn.Linear(in_features = n_cat_features, out_features = concat_number_of_features - 2*(concat_number_of_features//3))

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=concat_number_of_features*2)
        self.final_dense = nn.Linear(in_features=concat_number_of_features*2, out_features=1)

        

    def forward(self, whole_input):

        input1, input2, input3 = whole_input

        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.conv1(title_beg)
        after_acrivation1 = self.relu(title)
        final_title = self.adapt_avg_pool(after_acrivation1)

        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.conv1(full_beg)
        after_acrivation2 = self.relu(full)
        final_full = self.adapt_avg_pool(after_acrivation2)

        
        category = self.category(input3)


        
        concatenated = torch.cat(
            [
            final_title.view(final_title.size(0), -1),
            final_full.view(final_full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(concatenated)
        final_out = self.final_dense(out)

        
        return final_out