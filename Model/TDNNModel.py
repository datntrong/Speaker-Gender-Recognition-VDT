import torch.nn as nn
import torch.nn.functional as F
import torch

class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=40,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0,
            padding=0
    ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding

        self.kernel = nn.Conv1d(self.input_dim,
                                self.output_dim,
                                self.context_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)

        self.nonlinearity = nn.LeakyReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)

        x = self.kernel(x.transpose(1, 2))
        x = self.nonlinearity(x)
        x = self.drop(x)

        if self.batch_norm:
            x = self.bn(x)
        return x.transpose(1, 2)


class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals ** 2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor) / t)
        x = torch.cat([means, stds], dim=1)
        return x


class SoftMax(nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftMax, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = input
        W = self.W
        logits = F.linear(x, W)
        return logits


class XTDNN(nn.Module):

    def __init__(
            self,
            features_per_frame=40,
            final_features=1500,
            embed_features=512,
            dropout_p=0.0,
            batch_norm=True
    ):
        super(XTDNN, self).__init__()
        self.features_per_frame = features_per_frame
        self.final_features = final_features
        self.embed_features = embed_features
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        tdnn_kwargs = {'dropout_p': dropout_p, 'batch_norm': self.batch_norm}

        self.frame1 = TDNN(input_dim=self.features_per_frame, output_dim=512, context_size=5, dilation=1, **tdnn_kwargs)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2, **tdnn_kwargs)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3, **tdnn_kwargs)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame5 = TDNN(input_dim=512, output_dim=self.final_features, context_size=1, dilation=1, **tdnn_kwargs)

        self.tdnn_list = nn.Sequential(self.frame1, self.frame2, self.frame3, self.frame4, self.frame5)
        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(self.final_features * 2, self.embed_features)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embed_features)
        self.drop_embed = nn.Dropout(p=self.dropout_p)
        self.soft_max = SoftMax(num_features=embed_features, num_classes=6)

    def forward(self, x):
        x = self.tdnn_list(x)
        x = self.statspool(x)
        x = self.fc_embed(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        x = self.soft_max(x)
        return x