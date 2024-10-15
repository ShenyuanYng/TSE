import torch
import torch.nn as nn
from nets.normalization import get_norm_layer
from nets.conv_block import ConvBlock
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from nets.activation import get_activation_layer
from functools import partial
import random
from utils.audio_utils_torch import do_istft, do_stft, complex_multiply, scaling_magnitude


class FrameSeqAttn3U3NmUfdDualRNNBlockTFEmbViTLNFast(nn.Module):
    """
    Implementation of the intra-RNN and the inter-RNN
    input:
    """
    #def __getitem__(self, key):
    #    return getattr(self, key)

    def __init__(self,
            emb_dim, emb_ks, emb_hs, n_freqs, hidden_channels,
            n_head=4, attn_mode=0, approx_qk_dim=512,
            rnn_type='LSTM',
            act_type='elu', norm_type='LayerNormalization', convnormact_type=1,
            dropout=0., use_causal=0):
        super(FrameSeqAttn3U3NmUfdDualRNNBlockTFEmbViTLNFast, self).__init__()
        assert use_causal == 0
        assert emb_ks % emb_hs == 0
        assert norm_type == 'LayerNormalization'

        in_channels = emb_dim * emb_ks

        bidirectional = (use_causal == 0)
        self.intra_norm = get_norm_layer(emb_dim, norm_type, 3, use_causal=use_causal)
        self.intra_rnn = getattr(nn, rnn_type)(in_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        n_tmp = hidden_channels*2 if bidirectional else hidden_channels
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(n_tmp, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(n_tmp, emb_dim, emb_ks, stride=emb_hs, padding=0)  

        self.inter_norm = get_norm_layer(emb_dim, norm_type, 3, use_causal=use_causal)
        self.inter_rnn = getattr(nn, rnn_type)(in_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(n_tmp, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(n_tmp, emb_dim, emb_ks, stride=emb_hs, padding=0)

        cb_kwargs = {'norm_type': norm_type, 'act_type': act_type, 'convnormact_type': convnormact_type}
        self.convnormact_type = convnormact_type
        self.act_type = act_type
        self.norm_type = norm_type
        padding = 0

        E = math.ceil(approx_qk_dim*1.0/n_freqs) # approx_qk_dim is only approximate
        if n_freqs in [513,129,257,65,33,]:
            if n_freqs == 129:
                assert E in [4,8,]
            if n_freqs == 513:
                assert E in [2,]
            if n_freqs == 257:
                assert E in [2,4,]
            if n_freqs == 65:
                assert E in [8,]
            if n_freqs == 33:
                assert E in [16,]
            self.n_head = n_head
            assert emb_dim % n_head == 0
            if attn_mode == 4:
                for ii in range(n_head):
                    self.add_module('attn_conv_Q_%d'%ii, ConvBlock(emb_dim, E, (1,1), stride=(1,1), padding=padding,
                                act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
                    self.add_module('attn_conv_K_%d'%ii, ConvBlock(emb_dim, E, (1,1), stride=(1,1), padding=padding,
                                act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
                    self.add_module('attn_conv_V_%d'%ii, ConvBlock(emb_dim, emb_dim//n_head, (1,1), stride=(1,1), padding=padding,
                                act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
                self.add_module('attn_concat_proj', ConvBlock(emb_dim, emb_dim, (1,1), stride=(1,1), padding=padding,
                            act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
            elif attn_mode == 41:
                assert act_type == 'prelu' and  convnormact_type == 2
                self.add_module('attn_conv_Q', nn.Conv2d(emb_dim, n_head*E, 1, stride=1, padding=0))
                #self['attn_conv_Q'].bias.data[:] = 0.0
                self.attn_conv_Q.bias.data[:] = 0.0
                self.add_module('attn_norm_Q', get_norm_layer((n_head, E, n_freqs), 'AllHeadLayerNormalization4DCF', 4))

                self.add_module('attn_conv_K', nn.Conv2d(emb_dim, n_head*E, 1, stride=1, padding=0))
                #self['attn_conv_K'].bias.data[:] = 0.0
                self.attn_conv_K.bias.data[:] = 0.0
                self.add_module('attn_norm_K', get_norm_layer((n_head, E, n_freqs), 'AllHeadLayerNormalization4DCF', 4))

                self.add_module('attn_conv_V', nn.Conv2d(emb_dim, n_head*emb_dim//n_head, 1, stride=1, padding=0))
                #self['attn_conv_V'].bias.data[:] = 0.0
                self.attn_conv_V.bias.data[:] = 0.0
                self.add_module('attn_norm_V', get_norm_layer((n_head, emb_dim//n_head, n_freqs), 'AllHeadLayerNormalization4DCF', 4))

                self.add_module('attn_concat_proj', ConvBlock(emb_dim, emb_dim, (1,1), stride=(1,1), padding=padding,
                            act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
            else:
                raise
        else:
            raise

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        
    def forward(self, x):
        """
        x: [B, C, T, Q]
        out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        olp = self.emb_ks - self.emb_hs
        T = math.ceil((old_T+2*olp-self.emb_ks)*1./self.emb_hs)*self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q+2*olp-self.emb_ks)*1./self.emb_hs)*self.emb_hs + self.emb_ks

        x = x.permute(0,2,3,1) # [B, old_T, old_Q, C]
        x = F.pad(x, (0,0,olp,Q-old_Q-olp,olp,T-old_T-olp)) # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_) # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B*T, -1, self.emb_ks*C]) # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn) # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn) # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B*T, Q, C]) # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1,2) # [BT, C, Q]
            intra_rnn = F.unfold(intra_rnn[...,None], (self.emb_ks,1), stride=(self.emb_hs,1)) # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1,2) # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn) # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1,2) # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn) # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2,-1) # [B, T, Q, C]
        intra_rnn = intra_rnn + input_ # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1,2) # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_) # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B*Q, -1, self.emb_ks*C]) # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn) # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn) # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B*Q, T, C) # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1,2) # [BQ, C, T]
            inter_rnn = F.unfold(inter_rnn[...,None], (self.emb_ks,1), stride=(self.emb_hs,1)) # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1,2) # [BQ, -1, C*I] 

            inter_rnn, _ = self.inter_rnn(inter_rnn) # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1,2) # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn) # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2,-1) # [B, Q, T, C] 
        inter_rnn = inter_rnn + input_ # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0,3,2,1) # [B, C, T, Q]

        inter_rnn = inter_rnn[...,olp:olp+old_T,olp:olp+old_Q]
        batch = inter_rnn

        if hasattr(self, 'attn_conv_Q'):
            #Q = self['attn_norm_Q'](self['attn_conv_Q'](batch)) # [B, n_head, C, T, Q]
            #K = self['attn_norm_K'](self['attn_conv_K'](batch)) # [B, n_head, C, T, Q]
            #V = self['attn_norm_V'](self['attn_conv_V'](batch)) # [B, n_head, C, T, Q]
            Q = self.attn_norm_Q(self.attn_conv_Q(batch)) # [B, n_head, C, T, Q]
            K = self.attn_norm_K(self.attn_conv_K(batch)) # [B, n_head, C, T, Q]
            V = self.attn_norm_V(self.attn_conv_V(batch)) # [B, n_head, C, T, Q]
        else:
            raise
            all_Q, all_K, all_V = [], [], []
            for ii in range(self.n_head):
                #all_Q.append(self['attn_conv_Q_%d'%ii](batch)) # [B, C, T, Q]
                #all_K.append(self['attn_conv_K_%d'%ii](batch)) # [B, C, T, Q]
                #all_V.append(self['attn_conv_V_%d'%ii](batch)) # [B, C, T, Q]
                all_Q.append(getattr(self,'attn_conv_Q_%d'%ii)(batch)) # [B, C, T, Q]
                all_K.append(getattr(self,'attn_conv_K_%d'%ii)(batch)) # [B, C, T, Q]
                all_V.append(getattr(self,'attn_conv_V_%d'%ii)(batch)) # [B, C, T, Q]
            Q = torch.stack(all_Q, dim=1) # [B, n_head, C, T, Q]
            K = torch.stack(all_K, dim=1) # [B, n_head, C, T, Q]
            V = torch.stack(all_V, dim=1) # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:]) # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:]) # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:]) # [B*n_head, C, T, Q]

        """
        if callable(getattr(F, 'scaled_dot_product_attention', None)):
            Q = Q.transpose(1,2)
            Q = Q.flatten(start_dim=2) # [B', T, C*Q]

            K = K.transpose(1,2)
            K = K.flatten(start_dim=2) # [B', T, C*Q]

            V = V.transpose(1,2) # [B', T, C, Q]
            old_shape = V.shape
            V = V.flatten(start_dim=2) # [B', T, C*Q]
     
            V = F.scaled_dot_product_attention(Q, K, V) # [B', T, C*Q]
        else:
        """
        Q = Q.transpose(1,2)
        Q = Q.flatten(start_dim=2) # [B', T, C*Q]
        #
        K = K.transpose(2,3)
        K = K.contiguous().view([B*self.n_head, -1, old_T]) # [B', C*Q, T]
        #
        V = V.transpose(1,2) # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2) # [B', T, C*Q]
        emb_dim = Q.shape[-1]
        #
        #attn_mat = torch.matmul(Q, K.transpose(1,2)) / (emb_dim**0.5) # [B', T, T]
        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5) # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2) # [B', T, T]
        V = torch.matmul(attn_mat, V) # [B', T, C*Q]


        V = V.reshape(old_shape) # [B', T, C, Q]
        V = V.transpose(1,2) # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view([B,self.n_head*emb_dim,old_T,old_Q]) # [B, C, T, Q])
        #batch = self['attn_concat_proj'](batch) # [B, C, T, Q])
        batch = self.attn_concat_proj(batch) # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class DPRNNTF(nn.Module):

    #def __getitem__(self, key):
    #    return getattr(self, key)

    def __init__(self,
            n_srcs, n_freqs,
            n_imics, n_omics,
            seqmodel_type, n_layers, num_hid_units, emb_dim, emb_ks, emb_hs,
            masking_or_mapping, mask_nonlinearity, rmax,
            act_type='elu', norm_type='GlobalInstanceNormalization', convnormact_type=2,
            feature_type='ri',
            n_extra_feats=0,
            n_pred_components=2,
            first_conv_use_act=0,
            mag_sf=1.0,
            ):
        super(DPRNNTF, self).__init__()

        assert n_imics >= n_omics
        in_channels_list = [n_imics*n_pred_components] + [n_pred_components]*n_extra_feats

        t_ksize = 3
        ks, padding = (t_ksize,3), (t_ksize//2,1)
        for idx, in_channels in enumerate(in_channels_list):
            self.add_module('conv0-%d'%idx,
                    ConvBlock(in_channels,emb_dim,ks,padding=padding,
                        norm_type='GlobalLayerNormalization',act_type='None' if first_conv_use_act == 0 else act_type,convnormact_type=1))

        self.dual_rnn = nn.ModuleList([])
        if seqmodel_type in [3860, 3861,]:
            # fmseqattn34U3NmUfdLSTMViTLNallAllHeadFast
            for ii in range(n_layers):
                self.dual_rnn.append(
                        FrameSeqAttn3U3NmUfdDualRNNBlockTFEmbViTLNFast(emb_dim, emb_ks, emb_hs, n_freqs, num_hid_units,
                            n_head=4, attn_mode=41, approx_qk_dim=512 if seqmodel_type == 3860 else 1024,
                            rnn_type='LSTM', act_type=act_type, norm_type=norm_type, convnormact_type=convnormact_type))
        else:
            raise

        self.add_module('deconv0',
                ConvBlock(emb_dim,(n_srcs)*n_omics*n_pred_components,ks,
                    padding=padding,use_deconv=1,norm_type='None',act_type='None',convnormact_type=1))

        self.n_srcs = n_srcs
        self.n_freqs = n_freqs
        self.n_imics = n_imics
        self.n_omics = n_omics
        self.n_pred_components = n_pred_components
        self.seqmodel_type = seqmodel_type
        self.n_layers = n_layers
        self.num_hid_units = num_hid_units
        self.masking_or_mapping = masking_or_mapping
        self.mask_nonlinearity = mask_nonlinearity
        self.rmax = rmax
        self.t_ksize = t_ksize
        self.act_type = act_type
        self.norm_type = norm_type
        self.convnormact_type = convnormact_type
        self.n_extra_feats = n_extra_feats

        self.mag_sf = mag_sf

       
    def forward(self, batch, target_dict={}, current_epoch=None):
        # batch is [n_batch, n_pad_frames, 1, n_imics+n_extra_feats, n_pred_components, n_freqs]
        mix_ri = target_dict['mix_ri'][0] # [n_batch, n_pad_frames, 1, n_imics, n_pred_components, n_freqs]

        assert self.n_imics in [1, 8]
        mix_ri = mix_ri[:,:,:,:self.n_imics]
        n_batch, N = mix_ri.shape[:2]
        batch = mix_ri.contiguous()

        if hasattr(self, 'mag_sf') and self.mag_sf != 1.0:
            batch_cs = torch.complex(batch[...,0,:], batch[...,1,:])
            batch_r, batch_i, _ = scaling_magnitude(batch_cs, self.mag_sf, eps=1e-5)
            batch = torch.stack([batch_r, batch_i], dim=-2)

        batch = batch.view([n_batch,N,-1,self.n_pred_components*self.n_freqs])

        if self.n_extra_feats == 0:
            batch = batch.view([n_batch,N,-1,self.n_freqs])
            batch = batch.transpose(1,2) # [n_batch, -1, N, n_freqs]
            #batch = self['conv0-0'](batch)
            batch = getattr(self,'conv0-0')(batch)
        else:
            feats = [ batch ] + target_dict['extra_feats'] # [n_batch, N, -1, n_pred_components*n_freqs], and each [n_batch, N, 1, n_srcs, n_pred_components, n_freqs]
            batch = []
            for idx, feat in enumerate(feats):
                feat = feat.reshape([n_batch,N,-1,self.n_freqs])
                feat = feat.transpose(1,2) # [n_batch, -1, N, n_freqs]
                batch.append(getattr(self,'conv0-%d'%idx)(feat))
            batch = torch.sum(torch.stack(batch,dim=0),dim=0)

        for i in range(self.n_layers):
            batch = self.dual_rnn[i](batch)

        batch = self.deconv0(batch)
        # batch is [n_batch, -1, N, n_freqs]

        batch = batch.transpose(1,2) # [n_batch, N, -1, n_freqs]
        batch = batch.view([n_batch, N, self.n_srcs, self.n_omics, -1, self.n_freqs]) # -1 = self.n_pred_components + (0 or 1)

        masking_or_mapping = self.masking_or_mapping
        if masking_or_mapping == 0:
            # masking
            if self.mask_nonlinearity == 'linear':
                batch = torch.clamp(batch,-self.rmax,self.rmax)
            elif self.mask_nonlinearity == 'tanh':
                batch = torch.tanh(batch)*self.rmax
            elif self.mask_nonlinearity == 'linearNoT':
                pass    
            else:
                raise
        elif masking_or_mapping == 1:
            # mapping
            if self.mask_nonlinearity == 'linear':
                pass
            else:
                raise
        else:
            raise

        est_ri = batch
        # est_ri.shape is [n_batch, N, n_srcs, n_omics, n_pred_components, n_freqs]
        # mix_ri.shape is [n_batch, N,      1, n_imics, n_pred_components, n_freqs]
        if masking_or_mapping == 0:
            assert self.n_omics == 1
            est_r, est_i = est_ri[...,0,:], est_ri[...,1,:]         # each [n_batch, n_frames, n_srcs, 1, n_freqs]
            mix_r, mix_i = mix_ri[...,:1,0,:], mix_ri[...,:1,1,:]   # each [n_batch, n_frames,      1, 1, n_freqs]
            est_r, est_i = complex_multiply((est_r,est_i), (mix_r,mix_i))
            est_ri = torch.stack([est_r, est_i], dim=-2)            #      [n_batch, n_frames, n_srcs, 1, 2, n_freqs]
        elif masking_or_mapping == 1:
            pass
        else:
            raise

        return est_ri


class DPRNNTFStack(nn.Module):

    def __init__(self,
            n_srcs, n_freqs,
            n_imics, n_omics,
            seqmodel_type, n_layers, num_hid_units, emb_dim, emb_ks, emb_hs,
            masking_or_mapping, mask_nonlinearity, rmax,
            act_type='elu', norm_type='GlobalInstanceNormalization', convnormact_type=2,
            feature_type='ri',
            n_extra_feats=0,
            n_pred_components=2,
            first_conv_use_act=0,
            mag_sf=1.0,
            #
            use_pit_feat=0, use_lp_feat=None,
            n_layers_stack=0, n_iters=0,
            init_epochs_net1=0, init_epochs_net_stack=0,
            #
            ):
        super(DPRNNTFStack, self).__init__()

        self.n_srcs = n_srcs
 
        assert n_pred_components == 2
        assert n_omics == 1
        assert n_extra_feats == 0
        self.net1 = DPRNNTF(
                n_srcs, n_freqs,
                n_imics, n_omics,
                seqmodel_type, n_layers, num_hid_units, emb_dim, emb_ks, emb_hs,
                masking_or_mapping, mask_nonlinearity, rmax,
                act_type=act_type, norm_type=norm_type, convnormact_type=convnormact_type,
                feature_type=feature_type,
                n_extra_feats=n_extra_feats,
                n_pred_components=n_pred_components,
                first_conv_use_act=first_conv_use_act,
                mag_sf=mag_sf,
                )

        n_extra_feats = n_srcs*( (use_pit_feat>0) + (use_lp_feat is not None) )

        n_layers = n_layers_stack
        self.n_iters = n_iters
        self.iter_nets = nn.ModuleList([])
        for ii in range(self.n_iters):
            self.iter_nets.append(
                    DPRNNTF(
                        n_srcs, n_freqs,
                        n_imics, n_omics,
                        seqmodel_type, n_layers, num_hid_units, emb_dim, emb_ks, emb_hs,
                        masking_or_mapping, mask_nonlinearity, rmax,
                        act_type=act_type, norm_type=norm_type, convnormact_type=convnormact_type,
                        feature_type=feature_type,
                        n_extra_feats=n_extra_feats,
                        n_pred_components=n_pred_components,
                        first_conv_use_act=first_conv_use_act,
                        mag_sf=mag_sf,
                        )
                    )

        self.use_pit_feat = use_pit_feat
        self.use_lp_feat = use_lp_feat

        self.init_epochs_net1 = init_epochs_net1
        self.init_epochs_net_stack = init_epochs_net_stack

        self.n_imics = n_imics
        self.n_omics = n_omics


    def forward(self, batch, target_dict={}, current_epoch=100000000):
        init_epochs_1 = self.init_epochs_net1
        init_epochs_2 = self.init_epochs_net_stack

        assert len(target_dict['utt_id']) == 1
        mix_ri = target_dict['mix_ri'][0] # [n_batch, n_pad_frames, 1, n_imics, n_pred_components, n_freqs]

        batch = self.net1(None, target_dict) # [n_batch, N, n_srcs, n_omics, n_pred_components, n_freqs]
        assert batch.shape[3] == 1

        if current_epoch < init_epochs_1:
            tmp_n_iters = 0
        else:
            tmp_n_iters = min((current_epoch-init_epochs_1)//init_epochs_2 + 1, self.n_iters) if init_epochs_2 > 0 else self.n_iters

        for ii in range(tmp_n_iters):

            extra_feats = []

            if self.use_pit_feat > 0:
                pit_ri = batch.transpose(2,3) # [n_batch, N, 1, n_srcs, n_pred_components, n_freqs]
                extra_feats = extra_feats + [ pit_ri[:,:,:,cc:cc+1] for cc in range(pit_ri.shape[3]) ]
                #extra_feats = extra_feats + list( torch.split(pit_ri, 1, dim=3) )

            if self.use_lp_feat is not None:
                lp_func_name, lp_dict = parse_lp_str(self.use_lp_feat)

                if lp_func_name == 'ICP':
                    mix_cs = torch.complex(mix_ri[:,:,0,:,0], mix_ri[:,:,0,:,1]) # [n_batch, N, n_imics, n_freqs]
                    sph_cs = torch.complex(batch[:,:,:,0,0], batch[:,:,:,0,1]) # [n_batch, N, n_srcs, n_freqs]
                    ICP_cs = ICP_torch(mix_cs, sph_cs, **lp_dict,) # [n_batch, N, n_srcs, n_freqs]
                    ICP_cs = ICP_cs[:,:,None] # [n_batch, N, 1, n_srcs, n_freqs]
                    ICP_ri = torch.stack([torch.real(ICP_cs), torch.imag(ICP_cs)], dim=-2) # [n_batch, N, 1, n_srcs, n_pred_components, n_freqs]

                    extra_feats = extra_feats + [ ICP_ri[:,:,:,cc:cc+1] for cc in range(ICP_ri.shape[3]) ]

                elif lp_func_name == 'FCP':
                    mix_cs = torch.complex(mix_ri[:,:,0,:,0], mix_ri[:,:,0,:,1]) # [n_batch, N, n_imics, n_freqs]
                    sph_cs = torch.complex(batch[:,:,:,0,0], batch[:,:,:,0,1]) # [n_batch, N, n_srcs, n_freqs]

                    fcp_taps = lp_dict['taps']
                    fcp_delay = lp_dict['delay']
                    fcp_epsilon = lp_dict['epsilon']
                    fcp_weight_type = lp_dict['weight_type']
                    n_frames = mix_cs.shape[1]
                    frame_range_ = slice(0, n_frames, None) # updated on Feb. 22, 2024
                    # assuming first microphone is the reference microphone, do FCP at micorphone 1
                    inverse_power, _ = compute_inverse_power(mix_cs[:,frame_range_,:1], None, fcp_weight_type, fcp_epsilon) # [n_batch, n_frames, 1, n_freqs], [n_batch, 1, 1, 1]
                    FCP_cs = FCP_torch_2(
                            mix_cs[:,:,:1], # [n_batch, n_frames,      1, n_freqs]
                            sph_cs,         # [n_batch, n_frames, n_srcs, n_freqs]
                            frame_range_,
                            taps=fcp_taps, delay=fcp_delay, inverse_power=inverse_power,
                            )               # [n_batch, n_frames, n_srcs, 1, n_freqs]
                    FCP_cs = mix_cs[:,:,None,:1] - (FCP_cs - sph_cs[...,None,:])
                    FCP_cs = FCP_cs.transpose(2,3) # [n_batch, n_frames, 1, n_srcs, n_freqs]
                    FCP_ri = torch.stack([torch.real(FCP_cs), torch.imag(FCP_cs)], dim=-2) # [n_batch, N, 1, n_srcs, n_pred_components, n_freqs]

                    extra_feats = extra_feats + [ FCP_ri[:,:,:,cc:cc+1] for cc in range(FCP_ri.shape[3]) ]

                else:
                    raise

            target_dict['extra_feats'] = extra_feats
            batch = self.iter_nets[ii](None, target_dict) # [n_batch, N, n_srcs, n_omics, n_pred_components, n_freqs]

        return batch


def parse_lp_str(lp_str):
    lp_list = lp_str.split('_')
    lp_func_name = lp_list[0]
    lp_dict = {}
    lp_dict['taps'] = int(lp_list[1])
    lp_dict['delay'] = int(lp_list[2])
    if len(lp_list) > 3:
        lp_dict['epsilon'] = float(lp_list[3]) # for FCP
    if len(lp_list) > 4:
        lp_dict['weight_type'] = int(lp_list[4]) # for FCP
    return lp_func_name, lp_dict


def ICP_torch(
        mix_cs, sph_cs,
        taps=37, delay=3,
        diagload=1e-5):
    """ICP: linearly project mixture to estimate"""
    #
    # mix_cs.shape is [n_batch, n_frames, n_mics, n_freqs]
    # sph_cs.shape is [n_batch, n_frames, n_srcs, n_freqs]
    #
    n_batch, n_frames, n_mics, n_freqs = mix_cs.shape
    n_srcs = sph_cs.shape[-2]
    #
    mix_cs = mix_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_mics, n_frames]
    sph_cs = sph_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_srcs, n_frames]

    Y = sph_cs # [n_batch, n_freqs, n_srcs, n_frames]
    if delay == 0:
        Y_tilde = build_y_tilde_t(mix_cs, taps, delay) # [n_batch, n_freqs, taps*n_mics, n_frames]
    elif delay < 0:
        tmp_mix_cs = F.pad(mix_cs, (0,-delay)) # [n_batch, n_freqs, n_mics, n_frames+(-delay)]
        Y_tilde = build_y_tilde_t(tmp_mix_cs, taps, 0) # [..., taps*n_mics, n_frames+(-delay)]
        Y_tilde = Y_tilde[...,-delay:] # [n_batch, n_freqs, taps*n_mics, n_frames]
    else:
        raise

    R = torch.matmul(Y_tilde, Y_tilde.transpose(-1,-2).conj()) / n_frames # [n_batch, n_freqs, taps*n_mics, taps*n_mics]
    P = torch.matmul(Y_tilde, Y.transpose(-1,-2).conj()) / n_frames # [n_batch, n_freqs, taps*n_mics, n_srcs]

    old_dtype = R.dtype
    R = R.to(torch.complex128)
    P = P.to(torch.complex128)
    G = torch.linalg.solve(R+diagload*torch.eye(R.shape[-1],dtype=R.dtype,device=R.device), P) # [n_batch, n_freqs, taps*n_mics, n_srcs]
    G = G.to(old_dtype)

    ret = torch.matmul(Y_tilde.transpose(-1,-2), G.conj()) # [n_batch, n_freqs, n_frames, n_srcs]
    ret = ret.permute(0,2,3,1) # [n_batch, n_frames, n_srcs, n_freqs]
    return ret


#def build_y_tilde_t(x, taps, delay):
#    # x.shape [n_batch, n_freqs, n_mics, n_frames]
#    assert delay == 0
#    assert type(taps) is int
#    n_mics, n_frames = x.shape[-2:]
#    other_dims = list(x.shape[:-2])
#    x = x.view([-1,n_mics,n_frames,1])

#    x = F.pad(x, (0,0,taps-1,0)) # [-1,n_mics,taps-1+n_frames,1]
#    x_real = F.unfold(x.real, (taps,1)) # [-1, n_mics*taps, n_frames]
#    x_imag = F.unfold(x.imag, (taps,1)) # [-1, n_mics*taps, n_frames]
#    x = torch.complex(x_real, x_imag)

#    x = x.view([-1,n_mics,taps,n_frames])
#    x = x.transpose(1,2).contiguous() # [-1,taps,n_mics,n_frames]
#    x = x.view(other_dims+[taps*n_mics,n_frames])
#    return x
def build_y_tilde_t(x, taps, delay):
    # x.shape [n_batch, n_freqs, n_mics, n_frames]
    assert delay == 0
    assert type(taps) is int
    n_mics, n_frames = x.shape[-2:]
    other_dims = list(x.shape[:-2])
    x = x.view([-1,n_mics,n_frames,1])

    x = F.pad(x, (0,0,taps-1,0)) # [-1,n_mics,taps-1+n_frames,1]
    x = F.unfold(x, (taps,1)) # [-1, n_mics*taps, n_frames]

    x = x.view([-1,n_mics,taps,n_frames])
    x = x.transpose(1,2).contiguous() # [-1,taps,n_mics,n_frames]
    x = x.view(other_dims+[taps*n_mics,n_frames])
    return x


def FCP_torch_2(
        mix_cs, sph_cs, frame_range_,
        taps=20, delay=0, inverse_power=None,
        diagload=1e-5,
        ):
    """FCP: linearly project mixture to estimate"""
    #
    # mix_cs.shape is  [n_batch, n_frames, n_mics, n_freqs]
    # sph_cs.shape is  [n_batch, n_frames, n_srcs, n_freqs]
    # inverse_power is [n_batch, n_frames,      1, n_freqs]
    #
    n_batch, n_frames, n_mics, n_freqs = mix_cs.shape
    n_srcs = sph_cs.shape[-2]
    #
    mix_cs = mix_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_mics, n_frames]
    sph_cs = sph_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_srcs, n_frames]
    if inverse_power is not None:
        inverse_power = inverse_power.transpose(1,3) #  [n_batch, n_freqs,      1, n_frames]

    Y = mix_cs # [n_batch, n_freqs, n_mics, n_frames]
    if delay == 0:
        Y_tilde = build_y_tilde_t(sph_cs, taps, delay) # [n_batch, n_freqs, taps*n_srcs, n_frames]
    elif delay < 0:
        tmp_sph_cs = F.pad(sph_cs, (0,-delay)) # [n_batch, n_freqs, n_srcs, n_frames+(-delay)]
        Y_tilde = build_y_tilde_t(tmp_sph_cs, taps, 0) # [..., taps*n_srcs, n_frames+(-delay)]
        Y_tilde = Y_tilde[...,-delay:] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    elif delay > 0:
        tmp_sph_cs = F.pad(sph_cs, (delay,0)) # [n_batch, n_freqs, n_srcs, delay+n_frames]
        Y_tilde = build_y_tilde_t(tmp_sph_cs, taps, 0) # [..., taps*n_srcs, delay+n_frames]
        Y_tilde = Y_tilde[...,delay:] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    else:
        raise

    Y_tilde = Y_tilde[...,frame_range_] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    Y = Y[...,frame_range_] # [n_batch, n_freqs, n_mics, n_frames]
    n_frames = Y_tilde.shape[-1]

    Y_tilde = Y_tilde.view([n_batch, n_freqs, taps, n_srcs, n_frames])
    Y_tilde = Y_tilde.permute(0,1,3,4,2) # [n_batch, n_freqs, n_srcs, n_frames, taps]

    if inverse_power is not None:
        inverse_power = inverse_power[:,:,None]                    # [n_batch, n_freqs,      1,    1, n_frames]
        weighted_Y_tilde = Y_tilde.swapaxes(-1,-2) * inverse_power # [n_batch, n_freqs, n_srcs, taps, n_frames]
    else:
        weighted_Y_tilde = Y_tilde.swapaxes(-1,-2)                 # [n_batch, n_freqs, n_srcs, taps, n_frames]

    R = torch.matmul(weighted_Y_tilde, Y_tilde.conj()) / n_frames # [n_batch, n_freqs, n_srcs, taps, taps]

    Y = Y.swapaxes(-1, -2) # [n_batch, n_freqs, n_frames, n_mics]
    Y = Y[:,:,None] # [n_batch, n_freqs, 1, n_frames, n_mics]

    P = torch.matmul(weighted_Y_tilde, Y.conj()) / n_frames # [n_batch, n_freqs, n_srcs, taps, n_mics]

    old_dtype = R.dtype
    R = R.to(torch.complex128)
    P = P.to(torch.complex128)
    G = torch.linalg.solve(R+diagload*torch.eye(R.shape[-1],dtype=R.dtype,device=R.device), P) # [n_batch, n_freqs, n_srcs, taps, n_mics]
    G = G.to(old_dtype) 

    ret = torch.matmul(Y_tilde, G.conj()) # [n_batch, n_freqs, n_srcs, n_frames, n_mics]
    ret = ret.permute(0,3,2,4,1) # [n_batch, n_frames, n_srcs, n_mics, n_freqs]

    return ret


def compute_inverse_power(Y, max_=None, fcp_weight_type=0, fcp_epsilon=1e-3):
    """TF-unit weight in FCP"""
    # Y.shape is [n_batch, n_frames, n_imics, n_freqs]
    power = torch.mean(Y.real**2+Y.imag**2,dim=-2,keepdim=True) # [n_batch, n_frames, 1, n_freqs]
    if max_ is None:
        max_, _ = torch.max(power,dim=1,keepdim=True) # [n_batch, 1, 1, n_freqs]
        max_, _ = torch.max(max_,dim=3,keepdim=True) # [n_batch, 1, 1, 1]
    if fcp_weight_type == 0:
        return 1. / (power + max_*fcp_epsilon + 1e-6), max_ # [n_batch, n_frames, 1, n_freqs]
    elif fcp_weight_type == 1:
        raise
        return 1. / torch.maximum(power + 1e-6, max_*fcp_epsilon), max_ # [n_batch, n_frames, 1, n_freqs]
    elif fcp_weight_type == 2:
        raise
        return torch.ones_like(power), _ # [n_batch, n_frames, 1, n_freqs]
    else:
        raise
