#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


import transformers
transformers.__version__


# In[4]:


def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


# # TS regression
# ## - Time series only

# In[5]:


from transformers import PatchTSMixerConfig, PatchTSMixerForRegression
#from transformers import Trainer, TrainingArguments


# In[6]:


config_TS = PatchTSMixerConfig(context_length = 30, 
                            prediction_length = 1,
                            num_input_channels = 13, 
                            num_targets = 1,
                            patch_len = 10,
                            patch_stride = 5,
                            use_positional_encoding = True,
                            output_range=[0,1]
                           )
model_TS = PatchTSMixerForRegression(config_TS)


# In[7]:


model_TS.to('cuda')


# # TS-PD regression
# ## Time series & Point Distribution Fusion

# In[8]:


from transformers import PatchTSMixerConfig, PatchTSMixerModel


# In[9]:


from transformers import TimesformerConfig, TimesformerModel


# In[10]:


config_TS = PatchTSMixerConfig(context_length = 30, 
                            prediction_length = 1,
                            num_input_channels = 13, 
                            d_model = 48,
                            patch_len = 10,
                            patch_stride = 5,
                            use_positional_encoding = True,
                            ca_d_model = 128
                           )
model_TS = PatchTSMixerModel(config_TS)


# In[11]:


model_TS.to("cuda")


# In[12]:


calculate_model_size(model_TS)


# In[13]:


# dummy_TS
x = torch.zeros((32, 30, 13)).to('cuda')


# In[14]:


y = model_TS.forward(past_values=x)


# In[15]:


# last_hidden_state - torch.FloatTensor
# shape (batch_size, num_channels, num_patches, d_model)
y.last_hidden_state.shape


# In[16]:


# PD: Point distribution
config_PD = TimesformerConfig(image_size = 128,
                             patch_size = 8,
                             num_channels = 3,
                             num_frames = 4,
                             num_hidden_layers = 3,
                             num_attention_heads = 12,
                             hidden_size = 48,
                             intermediate_size = 256,
                             hidden_dropout_prob = 0.2)

model_PD = TimesformerModel(config_PD)


# In[17]:


model_PD.to("cuda")


# In[18]:


calculate_model_size(model_PD)


# In[19]:


# dummy_PD 
# pixel_values - torch.FloatTensor of 
# shape (batch_size, num_frames, num_channels, height, width)
x = torch.zeros((32, 4, 3, 96, 128)).to('cuda')


# In[20]:


y = model_PD.forward(pixel_values=x)


# In[ ]:





# In[21]:


# torch.FloatTensor of shape 
# (batch_size, sequence_length, hidden_size))
# (batch_size, 1 + H//P * W//P * T, hidden_size)
yy = y.last_hidden_state
yy.shape


# In[22]:


y[0][:,0].shape


# Sequence length 
# 
# = 1 + #frames(4) x H/8 x W/8
# 
# = 1 + 4 x 128/8 x 96/8
# 
# = 1 + 64 x 12
# 
# = 769
# 

# In[23]:


yy[:,1:, :].reshape([32, 192, 4, 48]).shape


# In[24]:


#x = torch.zeros((32, 30, 13)).to('cuda')


# In[25]:


#y = model.forward(past_values=x)


# In[26]:


#y


# In[27]:


#y.prediction_outputs.shape


# In[ ]:





# In[ ]:





# # Time-aware Cross-attention Module

# In[28]:


import torch
import torch.nn as nn
import math


# In[29]:


from transformers import PretrainedConfig


# In[30]:


class MultiModalCrossAttentionConfig(PretrainedConfig):
    def __init__(
        self,
        ## Time series specific configuration
        ts_context_length: int = 30,
        ts_patch_len: int = 5,
        ts_num_input_channels: int = 1,
        ts_patch_stride: int = 5,
        ts_d_model: int = 32,
        ts_time_step: int = 33*5,
        ## Point distribution configuration
        pd_num_frame: int = 4,
        pd_patch_size: int = 8,
        pd_height: int = 128,
        pd_width: int = 96,
        pd_d_model: int = 96,
        pd_time_step: int = 33*10,
        ## Time-preseved positional encoding
        #pe_parameter = 
        ## General cross attention configuration
        ca_d_model: int = 128,
        ca_num_head: int = 8,
        ca_dropout: float = 0.2,
        ca_num_layers: int = 3,
        # Classification/Regression configuration
        reg_d_fc: int = 128,
        reg_dropout: float = 0.2,
        **kwargs,
    ):
        self.ts_context_length = ts_context_length
        self.ts_patch_len = ts_patch_len
        self.ts_num_input_channels = ts_num_input_channels
        self.ts_patch_stride = ts_patch_stride
        self.ts_d_model = ts_d_model
        self.ts_time_step = ts_time_step
        ## Point distribution configuration
        self.pd_num_frame = pd_num_frame
        self.pd_patch_size = pd_patch_size
        self.pd_height = pd_height
        self.pd_width = pd_width
        self.pd_d_model = pd_d_model
        self.pd_time_step = pd_time_step
        ## Time-preseved positional encoding
        #pe_parameter = 
        ## General cross attention configuration
        self.ca_d_model = ca_d_model
        self.ca_num_head = ca_num_head
        self.ca_dropout = ca_dropout
        self.ca_num_layers = ca_num_layers
        # Classification/Regression configuration
        self.reg_d_fc = reg_d_fc
        self.reg_dropout = reg_dropout
        super().__init__(**kwargs)


# In[31]:


class TimeSeriesProjection(nn.Module):
    def __init__(self, config: MultiModalCrossAttentionConfig):
        super().__init__()
        self.num_channels = config.ts_num_input_channels
        self.ts_d_model = config.ts_d_model
        self.ca_d_model = config.ca_d_model
        
        # dimension of all channel's hidden state in a patch
        self.d_patch = self.num_channels * self.ts_d_model
        self.projection = nn.Linear(self.d_patch, self.ca_d_model)
        
        
    def forward(self, ts_hidden_state):
        # ts_hidden_state.shape
        # (batch_size, num_channels, num_patches, d_model)
        # num_patches is in time-axis
        batch_size = ts_hidden_state.shape[0]
        num_patches = ts_hidden_state.shape[2]
        ts_hidden_state = ts_hidden_state.permute(0,2,1,3).reshape(batch_size, 
                                                                   num_patches,
                                                                   self.d_patch)
        ts_hidden_state = self.projection(ts_hidden_state)
        return ts_hidden_state


# In[32]:


class PointDistProjection(nn.Module):
    def __init__(self, config: MultiModalCrossAttentionConfig):
        super().__init__()
        
        self.pd_num_frame = config.pd_num_frame
        self.patch_width = config.pd_width // config.pd_patch_size
        self.patch_height = config.pd_height // config.pd_patch_size
        self.pd_d_model = config.pd_d_model
        self.ca_d_model = config.ca_d_model
        
        # dimension of all patches' hidden state in a frame
        d_frame = self.patch_height*self.patch_width*self.pd_d_model
        self.projection = nn.Linear(d_frame, self.ca_d_model)
        
        
    def forward(self, pd_hidden_state):
        batch_size = pd_hidden_state.shape[0]
        patch_height, patch_width = self.patch_height, self.patch_width
        pd_num_frame = self.pd_num_frame
        pd_d_model = self.pd_d_model
        # pd_hidden_state.shape
        # (batch_size, 1 + H//P * W//P * T, hidden_size)
        # Drop the [CLS] token in the front
        hidden_state = pd_hidden_state[:,1:,:]
        # Time-preserved reshape
        #print("PD Projection Shape {}".format(hidden_state.shape))
        hidden_state = hidden_state.view(batch_size, 
                                         patch_height, patch_width,
                                         pd_num_frame, 
                                         pd_d_model
                                        ).reshape(
                                            batch_size,
                                            patch_height*patch_width,
                                            pd_num_frame,
                                            pd_d_model
                                         )
        hidden_state = hidden_state.permute(0, 2, 1, 3)
        hidden_state = hidden_state.reshape(batch_size, pd_num_frame, 
                                             patch_height*patch_width*pd_d_model)
        # Project to (batch_size, num_frame, ca_d_model)
        hidden_state = self.projection(hidden_state)
        return hidden_state


# In[33]:


class MultiModalPositionalEncoding(nn.Module):
    def __init__(self, config: MultiModalCrossAttentionConfig):
        super().__init__()
        self.pe_max_len = config.pe_max_len
        self.ca_d_model = config.ca_d_model
        
        max_len = self.pe_max_len
        d_model = self.ca_d_model
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.cos(position * div_term)
        self.encoding[:, 1::2] = torch.sin(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x, time_step=33):
        # time_step should be in millisecond
        # For 30 FPS, it will be 33 millisecond 
        sample_pos = torch.arange(x.size(1), device=x.device).float()*time_step
        self.encoding = self.encoding.to(x.device)
        print("sample_pos device: {}".format(sample_pos.get_device()))
        print("x device: {}".format(x.get_device()))
        print("encoding device: {}".format(self.encoding.get_device()))
        
        with torch.no_grad():
            encoding = self.encoding[:, sample_pos.long().to(x.device), :].expand(x.size(0), -1, -1)
        return encoding


# In[34]:


config = MultiModalCrossAttentionConfig(ts_context_length = 30,
                                        ts_patch_len = 5,
                                        ts_num_input_channels = 13,
                                        ts_patch_stride = 5,
                                        ts_d_model = 48,
                                        ts_time_step = 33*5,
                                        # PD parameter
                                        pd_d_model = 48,
                                        pd_time_step = 33*10,
                                        pe_max_len = 1000, 
                                        # CA
                                        ca_d_model = 48, 
                                        ca_num_head = 8,
                                        ca_num_layers = 3
                                       )


# In[35]:


class CrossAttentionLayer(nn.Module):
    def __init__(self, config: MultiModalCrossAttentionConfig):
        super().__init__()
        self.ca_d_model = config.ca_d_model
        d_model = self.ca_d_model
        self.ca_num_head = config.ca_num_head
        num_head = self.ca_num_head
        self.ca_dropout = config.ca_dropout
        dropout = self.ca_dropout
        # Attention
        self.pd_attention = nn.MultiheadAttention(d_model, num_head, dropout=dropout, batch_first=True)
        self.ts_attention = nn.MultiheadAttention(d_model, num_head, dropout=dropout, batch_first=True)
        # Separate linear layers
        self.pd_linear = nn.Linear(d_model, d_model)
        self.ts_linear = nn.Linear(d_model, d_model)
        # Normalization layers
        self.pd_norm = nn.LayerNorm(d_model)
        self.ts_norm = nn.LayerNorm(d_model)
        
        self.pd_dropout = nn.Dropout(dropout)
        self.ts_dropout = nn.Dropout(dropout)
        
        
    def forward(self, pd_hs, ts_hs):
        # point distribution to time series cross attention
        pd_to_ts, _ = self.pd_attention(pd_hs, ts_hs, ts_hs)
        # time series to point distribution cross attention
        ts_to_pd, _ = self.ts_attention(ts_hs, pd_hs, pd_hs)

        # linearly transform the outputs
        pd_hs = self.pd_linear(pd_to_ts)
        pd_hs = self.pd_dropout(pd_hs)
        pd_hs = self.pd_norm(pd_hs)
        
        ts_hs = self.ts_linear(ts_to_pd)
        ts_hs = self.ts_dropout(ts_hs)
        ts_hs = self.ts_norm(ts_hs)

        return pd_hs, ts_hs


# In[36]:


class MultiModalCrossAttentionRegressor(nn.Module):
    def __init__(self, config: MultiModalCrossAttentionConfig):
        super().__init__()        
        num_layers = config.ca_num_layers
        ca_d_model = config.ca_d_model
        reg_d_fc = config.reg_d_fc
        reg_dropout = config.reg_dropout
        
        # Time-preserved projection layers
        self.pd_proj = PointDistProjection(config)
        self.ts_proj = TimeSeriesProjection(config)
        # Postional encoding
        self.ts_time_step = config.ts_time_step
        self.pd_time_step = config.pd_time_step
        self.pos_encoding = MultiModalPositionalEncoding(config)
        # Attention|
        self.ca_layers = nn.ModuleList([CrossAttentionLayer(config) for _ in range(num_layers)])
        
        self.fc1 = nn.Linear(ca_d_model*2, reg_d_fc)
        self.fc_dropout = nn.Dropout(reg_dropout)
        self.fc2 = nn.Linear(reg_d_fc, 1)
    
        
    def forward(self, pd_hidden_state, ts_hidden_state):
        # Project to the same dimension with time-preserve
        pd_hs = self.pd_proj(pd_hidden_state)
        ts_hs = self.ts_proj(ts_hidden_state)
        
        # Generate positional encodings
        pd_pe = self.pos_encoding(pd_hs, self.pd_time_step)
        ts_pe = self.pos_encoding(ts_hs, self.ts_time_step)
        
        # Add positional encodings
        pd_hs = pd_hs + pd_pe
        ts_hs = ts_hs + ts_pe
        
        for layer in self.ca_layers:
            pd_hs, ts_hs = layer(pd_hs, ts_hs)
            
        # Concatenating the mean of each stream
        # The mean pool eliminates/reduces the temporal axis(dim=1)
        pd_hs_mean = pd_hs.mean(dim=1)
        ts_hs_mean = ts_hs.mean(dim=1)
        combined = torch.cat([pd_hs_mean, ts_hs_mean], dim=-1)
        # combined.shape = [bs, d_modelx2]
        out = self.fc1(combined)
        out = nn.functional.gelu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        
        return out


# In[37]:


ts_hs = torch.zeros([32, 13, 5, 48])
pd_hs = torch.zeros([32, 769, 48])


# In[38]:


MMCARegressor =  MultiModalCrossAttentionRegressor(config)


# In[39]:


y = MMCARegressor(pd_hs, ts_hs)


# # DeepSEE Model

# In[40]:


from transformers import PatchTSMixerConfig, PatchTSMixerModel
from transformers import TimesformerConfig, TimesformerModel


# In[41]:


# PD: Point distribution
config_PD = TimesformerConfig(image_size = 128,
                             patch_size = 8,
                             num_channels = 3,
                             num_frames = 4,
                             num_hidden_layers = 3,
                             num_attention_heads = 12,
                             hidden_size = 48,
                             intermediate_size = 256,
                             hidden_dropout_prob = 0.2)


# In[42]:


config_TS = PatchTSMixerConfig(context_length = 30, 
                            prediction_length = 1,
                            num_input_channels = 13, 
                            d_model = 48,
                            patch_len = 10,
                            patch_stride = 5,
                            use_positional_encoding = True,
                            ca_d_model = 128
                           )


# In[43]:


config_CA = MultiModalCrossAttentionConfig(ts_context_length = 30,
                                          ts_patch_len = 5,
                                          ts_num_input_channels = 13,
                                          ts_patch_stride = 5,
                                          ts_d_model = 48,
                                          ts_time_step = 33*5,
                                          # PD parameter
                                          pd_d_model = 48,
                                          pd_time_step = 33*10,
                                          pe_max_len = 1000, 
                                          # CA
                                          ca_d_model = 48, 
                                          ca_num_head = 8,
                                          ca_num_layers = 3
                                         )


# In[44]:


class DeepSEEModel(nn.Module):
    def __init__(self,
                 pd_config: TimesformerConfig,
                 ts_config: PatchTSMixerConfig,
                 ca_config: MultiModalCrossAttentionConfig):
        super().__init__()        
        
        self.pd_config = pd_config
        self.ts_config = ts_config
        self.ca_config = ca_config
        
        self.pd_encoder = TimesformerModel(pd_config)
        self.ts_encoder = PatchTSMixerModel(ts_config)
        
        self.ca_regressor = MultiModalCrossAttentionRegressor(ca_config)
    
        
    def forward(self, point_dist, time_series):
        
        pd_hs = self.pd_encoder(point_dist).last_hidden_state
        ts_hs = self.ts_encoder(time_series).last_hidden_state
        
        return self.ca_regressor(pd_hs, ts_hs)


# In[45]:


# dummy_TS
x_ts = torch.zeros((32, 30, 13)).to('cuda')
x_pd = torch.zeros((32, 4, 3, 96, 128)).to('cuda')


# In[46]:


deepSEEModel = DeepSEEModel(config_PD, config_TS, config_CA).to('cuda')


# In[47]:


y = deepSEEModel(x_pd, x_ts)


# In[48]:


y.shape


# In[49]:


calculate_model_size(deepSEEModel)


# In[ ]:




