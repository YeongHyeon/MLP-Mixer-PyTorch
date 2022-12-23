import torch
import torch.nn as nn
import torch.nn.functional as F

class Neuralnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.who_am_i = "MLP-Mixer"

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.patch_size = kwargs['patch_size']
        self.dim_emb = kwargs['dim_emb']
        self.d_mix_t = kwargs['d_mix_t']
        self.d_mix_c = kwargs['d_mix_c']
        self.depth = kwargs['depth']

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.ngpu = kwargs['ngpu']
        self.device = kwargs['device']

        self.dim_s = (self.dim_h//self.patch_size)*(self.dim_w//self.patch_size)

        self.params, self.names = [], []
        self.params.append(Embedding(self.dim_c, self.dim_emb, self.patch_size, stride=self.patch_size, name='embedding').to(self.device))
        self.names.append("embedding")

        for idx_depth in range(self.depth):
            self.params.append(MixBlock(self.dim_s, self.d_mix_t, name='mix_token_%d' %(idx_depth)).to(self.device))
            self.names.append("mix_token_%d" %(idx_depth))
            self.params.append(MixBlock(self.dim_emb, self.d_mix_c, name='mix_channel_%d' %(idx_depth)).to(self.device))
            self.names.append("mix_channel_%d" %(idx_depth))

        self.params.append(Classifier(self.dim_s, self.num_class, name='classifier').to(self.device))
        self.names.append("classifier")
        self.modules = nn.ModuleList(self.params)

    def forward(self, x):

        for idx_param, _ in enumerate(self.params):
            if(("mix_channel" in self.names[idx_param]) or ("classifier" in self.names[idx_param])):
                x = torch.permute(x, (0, 2, 1))
            x = self.params[idx_param](x)
            if("mix_channel" in self.names[idx_param]):
                x = torch.permute(x, (0, 2, 1))
        y_hat = x

        return {'y_hat':y_hat}

    def loss(self, dic):

        y, y_hat = dic['y'], dic['y_hat']
        loss_ce = nn.CrossEntropyLoss()
        opt_b = loss_ce(y_hat, target=y)
        opt = torch.mean(opt_b)

        return {'opt': opt}

class Embedding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, name=""):
        super().__init__()
        self.emb = nn.Sequential()
        self.emb.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0))

    def forward(self, x):

        x = self.emb(x)
        [n, c, h, w] = x.shape
        return torch.reshape(x, (n, c, h*w))

class MixBlock(nn.Module):

    def __init__(self, in_channels, mix_channels, name=""):
        super().__init__()
        self.mix = nn.Sequential()
        self.mix.add_module("%s_ln0" %(name), nn.LayerNorm(in_channels))
        self.mix.add_module("%s_lin0" %(name), nn.Linear(in_channels, mix_channels))
        self.mix.add_module("%s_act0" %(name), nn.GELU())
        self.mix.add_module("%s_lin1" %(name), nn.Linear(mix_channels, in_channels))

    def forward(self, x):

        x_mix = self.mix(x)
        return x + x_mix

class Classifier(nn.Module):

    def __init__(self, in_channels, out_channels, name=""):
        super().__init__()
        self.clf = nn.Sequential()
        self.clf.add_module("%s_lin0" %(name), nn.Linear(in_channels, out_channels))

    def forward(self, x):

        gap = torch.mean(x, axis=-1)
        return self.clf(gap)
