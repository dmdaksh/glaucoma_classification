import sys
import numpy as np
import torch
from torch import nn
from monai.transforms import GridPatch
from monai.networks.blocks import TransformerBlock
from torchvision.utils import make_grid
from torch.nn import functional as F
from torchvision.transforms import ToPILImage
from math import sqrt

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_2d_sincos_emb(self, hidden_size, grid_size, cls_token=True):
        grid_h = torch.arange(0, grid_size)
        grid_w = torch.arange(0, grid_size)
        grid = torch.meshgrid(grid_w, grid_h)
        grid = torch.stack(grid, dim=0) # (2, grid_w, grid_h)

        grid = grid.view(2, 1, grid_size, grid_size) # (2, 1, grid_w, grid_h)
        pos_emb = self.get_2d_sincos_pos_embed_from_grid(hidden_size, grid)
        if cls_token:
            pos_emb = torch.cat([torch.zeros(1, hidden_size), pos_emb], dim=0)
        return pos_emb
    
    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = torch.cat([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.view(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward(self, hidden_size, grid_size, cls_token=True):
        return self.get_2d_sincos_emb(hidden_size, grid_size, cls_token)

class Encoder(nn.Module):
    def __init__(self, hidden_size=768, mlp_dim=3072, num_heads=4, num_layers=8, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, patch_embeddings):
        x = patch_embeddings
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_channels=3, patch_size=5, num_heads=4, mlp_dim=3072, num_layers=8, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(in_features=hidden_dim, out_features=out_channels * patch_size**2)

    def forward(self, patch_embeddings):

        b = patch_embeddings.shape[0]
        n_patches = patch_embeddings.shape[1]

        x = patch_embeddings # (None, n_patches, feat_dim)

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # output layer
        out = self.out(x) # (None, n_patches, n_channels*patch_size**2)

        # reshape output into image
        out = out.view(b, n_patches, self.out_channels, self.patch_size, self.patch_size) # (None, n_patches, n_channels, patch_size, patch_size)
        return out

class MAE(nn.Module):
    def __init__(self, spatial_dim, n_channels, mask_ratio=0.75, hidden_size=768, decoder_hidden_size=768, patch_size=5, mlp_dim=3072, decoder_mlp_dim=3072, num_heads=4, decoder_num_heads=4, num_layers=8, decoder_num_layers=8, dropout_rate=0.0, decoder_dropout_rate=0.0):
        super(MAE, self).__init__()
        self.spatial_dim = spatial_dim
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_size
        self.decoder_hidden_dim = decoder_hidden_size
        self.mask_ratio = mask_ratio
        self.n_patches = (self.spatial_dim // self.patch_size) * (self.spatial_dim // self.patch_size)
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.decoder_num_heads = decoder_num_heads
        self.decoder_mlp_dim = decoder_mlp_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_dropout_rate = decoder_dropout_rate

        # initialize component models
        self.patch_embedding = nn.Linear(in_features=self.n_channels * self.patch_size * self.patch_size, out_features=self.hidden_dim)
        self.decoder_embedding = nn.Linear(in_features=self.hidden_dim, out_features=self.decoder_hidden_dim)
        self.encoder = Encoder(self.hidden_dim, self.mlp_dim, self.num_heads, self.num_layers, self.dropout_rate)
        self.decoder = Decoder(self.decoder_hidden_dim, self.n_channels, self.patch_size, self.decoder_num_heads, self.decoder_mlp_dim, self.decoder_num_layers, self.decoder_dropout_rate) 

        # initialize pos embedding and mask_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_dim))
        self.pos_embedding = nn.Parameter(PositionalEncoding()(self.hidden_dim, int(sqrt(self.n_patches)), cls_token=True).unsqueeze(0), requires_grad=False)
        self.decoder_pos_embedding = nn.Parameter(PositionalEncoding()(self.decoder_hidden_dim, int(sqrt(self.n_patches)), cls_token=True).unsqueeze(0), requires_grad=False)

    def extract_patches(self, x):
        patches = GridPatch(patch_size=(x.shape[1], self.patch_size, self.patch_size))(x).to(x.device)
        patches = torch.permute(patches, (1, 0, 2, 3, 4))
        return patches
    
    def combine_patches(self, x):
        image = make_grid(x, nrow=self.spatial_dim // self.patch_size, padding=0, normalize=False)
        return image
    
    def random_sample_tokens(self, x):
        b = x.shape[0]

        # shuffle indices for each sample: (None, n_patches)
        shuffled_indices = torch.stack([torch.randperm(self.n_patches) for _ in range(b)], dim=0).to(x.device)

        # split indices into visible and masked sets
        visible_idxs = shuffled_indices[:, :int((1-self.mask_ratio)*self.n_patches)]
        mask_idxs = shuffled_indices[:, int((1-self.mask_ratio)*self.n_patches):]

        # extract the visible and masked indices
        visible_tokens = x.gather(dim=1, index=visible_idxs[:, :, None].repeat(1, 1, self.hidden_dim))
        return visible_tokens, visible_idxs, mask_idxs
    
    def weighted_sample_tokens(self, x, fundus_masks):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        fundus_masks: [N, 1, H, W] 
        """
        
        b = x.shape[0]  # batch, length (# patches), dim
        
        # downsample fundus masks
        dim = int(self.n_patches**0.5)
        fundus_masks_downsampled = F.interpolate(fundus_masks, size=(dim, dim)) # (N, 1, dim, dim)
        fundus_masks_flat = fundus_masks_downsampled.view(b, -1) # (N, L)

        visible_idxs = []
        mask_idxs = []
        for i in range(fundus_masks_flat.shape[0]):
            
            # get background indices - these will be part of the set of visible patches
            background_idxs = torch.argwhere(fundus_masks_flat[i] == 0).squeeze() # ((1-k)% * L, )

            # get the number of visible patches remaining to reach (1-mask_ratio) %
            # these remaining visible patches will be sampled from the foreground idxs
            n_visible_remaining = int((1 - self.mask_ratio)*self.n_patches - background_idxs.shape[0])

            # get foreground indices
            foreground_idxs = torch.argwhere(fundus_masks_flat[i]).squeeze() # (k% * L,) 

            # randomly sample indices
            n_foreground = foreground_idxs.shape[0] # k% * L
            shuffled_indices = torch.gather(foreground_idxs, dim=0, index=torch.randperm(n_foreground, device=x.device)) # k% * L
            
            # get remaining visible indices and store
            visible_idxs_remaining = shuffled_indices[:n_visible_remaining]
            visible_idxs_i = torch.cat([background_idxs, visible_idxs_remaining])
            visible_idxs.append(visible_idxs_i)
            
            # store masked idxs
            mask_idxs_i = shuffled_indices[n_visible_remaining:]
            mask_idxs.append(mask_idxs_i)

            print(background_idxs.shape, foreground_idxs.shape, visible_idxs_i.shape, mask_idxs_i.shape)

        # this is the set of visible and masked indices inside the foreground
        visible_idxs = torch.stack(visible_idxs, dim=0)
        mask_idxs = torch.stack(mask_idxs, dim=0)

        # extract visible and masked tokens
        visible_tokens = torch.gather(x, dim=1, index=visible_idxs.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
        return visible_tokens, visible_idxs, mask_idxs

    def forward_encode(self, x_patches, fundus_masks=None, mask_patches=True):

        # Create patch embedding: (None, n_patches, H)
        x_emb = self.patch_embedding(x_patches)

        # add positional embedding: (None, n_patches, H)
        x_emb = x_emb + self.pos_embedding[:, 1:, :]

        # Sample embeddings / image patches: (None, (1-r)*n_patches, H), r = [0, 1]
        if mask_patches:
            visible_tokens, visible_idxs, masked_idxs = self.random_sample_tokens(x_emb) if fundus_masks is None else self.weighted_sample_tokens(x_emb, fundus_masks)
        else:
            visible_tokens = x_emb
            visible_idxs = torch.arange(0, self.n_patches)
            masked_idxs = torch.tensor([])

        # append cls token: (None, 1 + (1-r)*n_patches, H), r = [0, 1]
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_token = cls_token.repeat(visible_tokens.shape[0], 1, 1)
        visible_tokens = torch.cat([cls_token, visible_tokens], dim=1)

        # apply encoder model on visible tokens: (None, 1 + (1-r)*n_patches, H), r = [0, 1]
        x_patches_encoded = self.encoder(visible_tokens)

        return x_patches_encoded, visible_idxs, masked_idxs

    def forward_decode(self, inputs, visible_idxs, masked_idxs):

        # Input: (None, 1 + n_patches*(1-r), H)
        x = inputs
        b = x.shape[0]

        # embed encodings # (None, 1 + n_patches*(1-r), H')
        x_emb = self.decoder_embedding(x)

        # attach mask tokens: (None, 1+n_patches, H')
        mask_tokens = self.mask_token.repeat(b, masked_idxs.shape[1], 1) # int(self.mask_ratio * self.n_patches)
        x_plus_mask_emb = torch.cat([x_emb[:, 1:, :], mask_tokens], dim=1) # (None, 1 + n_patches, H)

        # perform inverse shuffle on tokens: (None, 1+n_patches, H)
        shuffled_idxs = torch.cat([visible_idxs, masked_idxs], dim=1)
        sorted_idxs = torch.argsort(shuffled_idxs, dim=1)
        x_plus_mask_emb = x_plus_mask_emb.gather(dim=1, index=sorted_idxs[:, :, None].repeat(1, 1, self.decoder_hidden_dim))
        x_plus_mask_emb = torch.cat([x_emb[:, :1, :], x_plus_mask_emb], dim=1) # add cls token

        # add positional embedding
        x_plus_mask_emb = x_plus_mask_emb + self.decoder_pos_embedding

        # apply decoder network: (None, n_patches+1, C, P, P)
        x_patches_pred = self.decoder(x_plus_mask_emb)

        # remove cls token
        x_patches_pred = x_patches_pred[:, 1:, :]
        return x_patches_pred

    def forward(self, inputs, fundus_masks=None, mask_patches=True, test_mode=False, encode_only=False):

        # inputs: (None, C, H, W)
        x = inputs
        b = x.shape[0]

        # Split image into patches: (None, n_patches, C, P, P)
        x_patches = self.extract_patches(x)
        # Unravel patches into vectors: (None, n_patches, C*P*P)
        x_patches = x_patches.view(b, self.n_patches, -1)

        # perform forward pass through encoder network: (None, 1 + n_patches * (1 - ratio), H), (None, (1 - ratio) * n_patches), (None, ratio * n_patches)
        x_visible_patches_encoded, visible_idxs, masked_idxs = self.forward_encode(x_patches, fundus_masks, mask_patches)

        if encode_only:
            return x_visible_patches_encoded[:, 1:, :]

        # perform forward pass through decoder network: (None, n_patches, C, P, P)
        x_patches_pred = self.forward_decode(x_visible_patches_encoded, visible_idxs, masked_idxs)
        # get predictions for masked patches only: (None, ratio*n_patches, C, P, P)
        masked_idxs = torch.sort(masked_idxs)[0]
        masked_patches_pred = x_patches_pred.gather(dim=1, index=masked_idxs[:, :, None, None, None].repeat(1, 1, self.n_channels, self.patch_size, self.patch_size))
        masked_patches_true = x_patches.gather(dim=1, index=masked_idxs[:, :, None].repeat(1, 1, self.n_channels*self.patch_size*self.patch_size))
        masked_patches_true = masked_patches_true.view(b, masked_idxs.shape[1], self.n_channels, self.patch_size, self.patch_size)
        # int(self.mask_ratio*self.n_patches)
        
        # covert all patches into images 
        x_patches = x_patches.view(b, self.n_patches, self.n_channels, self.patch_size, self.patch_size)
        mask = torch.ones_like(x_patches) * 0.5
        x_patches_masked = torch.scatter(x_patches, dim=1, src=mask, index=masked_idxs[:, :, None, None, None].repeat(1, 1, self.n_channels, self.patch_size, self.patch_size))
    
        # TEMPORARY: calculate mse between all patches
        # masked_patches_pred, masked_patches_true = x_patches_pred.clone(), x_patches.clone()

        if test_mode:
            return masked_patches_pred, masked_patches_true, x_patches, x_patches_masked, x_patches_pred

        return masked_patches_pred, masked_patches_true

    def test(self):

        torch.manual_seed(1234)

        # run model on inputs
        from PIL import Image
        from torchvision.transforms import Resize
        from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor

        def load_image(im):
            x = LoadImage(image_only=True)(im)
            x = EnsureChannelFirst()(x)
            x = Resize((self.spatial_dim, self.spatial_dim))(x)
            x = ScaleIntensity()(x)
            x = ToTensor()(x)
            return x

        x = torch.stack(list(map(load_image, ['images/fundus_im_0.png', 'images/fundus_im_1.png', 'images/fundus_im_2.png'])), dim=0)
        msk = torch.stack(list(map(load_image, ['images/fundus_mask_0.png', 'images/fundus_mask_1.png', 'images/fundus_mask_2.png'])), dim=0)
        # print(x.shape, msk.shape)
        # sys.exit(0)
        _, _, x_patches_true, x_patches_true_masked, x_patches_pred = self.forward(x, fundus_masks=msk, test_mode=True) # (None, n_patches, C, patch_size, patch_size)
        assert x_patches_pred.shape == x_patches_true.shape, (x_patches_pred.shape, x_patches_true.shape)
        
        # reconstruct images from patches and visualize
        for i in range(x.shape[0]):
            true_image = self.combine_patches(x_patches_true[i])
            true_image_masked = self.combine_patches(x_patches_true_masked[i])
            pred_image = self.combine_patches(x_patches_pred[i])
            
            grid = make_grid([true_image, true_image_masked, pred_image], nrow=3)
            grid = ToPILImage(mode='RGB')(grid)
            grid.save(f'images/test_grid_{i}.png')

        print('Test Successful!')


class MAELinearProbing(nn.Module):
    def __init__(self, mae_config, pretrained_weights_path, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.mae_pretrained = MAE(**mae_config)
        self.mae_pretrained.load_state_dict(torch.load(pretrained_weights_path))
        # freezing pretrained model
        for param in self.mae_pretrained.parameters():
            param.requires_grad = False
        self.batch_norm = nn.BatchNorm1d(self.mae_pretrained.n_patches)
        self.fc = nn.Sequential(
            nn.Linear(self.mae_pretrained.n_channels*self.mae_pretrained.patch_size**2, self.n_classes*128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.n_classes*128, self.n_classes),
        )
    
    def forward(self, x):
        x = self.mae_pretrained(x, mask_patches=False, encode_only=True)
        x = self.batch_norm(x)
        # global avg pooling
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

class MAEFineTuning(nn.Module):
    def __init__(self, mae_config, pretrained_weights_path, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.mae_pretrained = MAE(**mae_config)
        self.mae_pretrained.load_state_dict(torch.load(pretrained_weights_path))
        # freezing pretrained model
        for param in self.mae_pretrained.parameters():
            param.requires_grad = False
        self.batch_norm = nn.BatchNorm1d(self.mae_pretrained.n_patches)

        self.fc = nn.Sequential(
            nn.Linear(self.mae_pretrained.n_channels*self.mae_pretrained.patch_size**2, self.n_classes*128),
            nn.ReLU,
            nn.Dropout(0.5),
            nn.Linear(self.n_classes*128, self.n_classes*8),
            nn.ReLU,
            nn.Dropout(0.5),
            nn.Linear(self.n_classes*8, self.n_classes),
        )
    
    def forward(self, x):
        x = self.mae_pretrained(x, mask_patches=False, encode_only=True)
        x = self.batch_norm(x)
        # global avg pooling
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x





if __name__ == '__main__':
    model = MAE(spatial_dim=512, n_channels=3, mask_ratio=0.75, patch_size=32)
    model.test()