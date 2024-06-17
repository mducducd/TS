import torch.nn as nn
from .wavenet import WaveNet

class Audio2Feature(nn.Module):
    def __init__(self, feature_decoder='WaveNet', time_frame_length=8, A2L_GMM_ndim=25*3, A2L_GMM_ncenter=1, predict_length=90,
                 APC_hidden_size=512, loss = 'L2', output_length = 90):
        super(Audio2Feature, self).__init__()
        # self.opt = opt
        A2L_wavenet_input_channels = APC_hidden_size
        self.feature_decoder = feature_decoder
        if loss == 'GMM':
            output_size = (2 * A2L_GMM_ndim + 1) * A2L_GMM_ncenter
        elif loss == 'L2':
            num_pred = predict_length
            output_size = A2L_GMM_ndim * num_pred
        # define networks
        if feature_decoder == 'WaveNet':
            self.WaveNet = WaveNet(residual_layers = 10,
                                   residual_blocks = 3,
                                   dilation_channels = 32,
                                   residual_channels = 32,
                                   skip_channels = 256,
                                   kernel_size = 2,
                                #    output_length = output_length,
                                   use_bias = False,
                                   cond = False,
                                   input_channels = 45,
                                   ncenter = 1,
                                   ndim = 73*2,
                                   output_channels = 73,
                                   cond_channels = 256,
                                   activation = 'leakyrelu')
            self.item_length = self.WaveNet.receptive_field + time_frame_length - 1
        elif feature_decoder == 'LSTM':
            self.downsample = nn.Sequential(
                    nn.Linear(in_features=APC_hidden_size * 2, out_features=APC_hidden_size),
                    nn.BatchNorm1d(APC_hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(APC_hidden_size, APC_hidden_size),
                    )
            self.LSTM = nn.LSTM(input_size=APC_hidden_size,
                                hidden_size=256,
                                num_layers=3,
                                dropout=0,
                                bidirectional=False,
                                batch_first=True)
            self.fc = nn.Sequential(
                    nn.Linear(in_features=256, out_features=512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, output_size))
                    
    
    def forward(self, audio_features, output_length):
        '''
        Args:
            audio_features: [b, T, ndim]
        '''
        if self.feature_decoder == 'WaveNet':
            pred = self.WaveNet.forward(audio_features.permute(0,2,1), output_length) 
        elif self.feature_decoder == 'LSTM':
            bs, item_len, ndim = audio_features.shape
            # new in 0324
            audio_features = audio_features.reshape(bs, -1, ndim*2)
            down_audio_feats = self.downsample(audio_features.reshape(-1, ndim*2)).reshape(bs, int(item_len/2), ndim)
            output, (hn, cn) = self.LSTM(down_audio_feats)
#            output, (hn, cn) = self.LSTM(audio_features)
            pred = self.fc(output.reshape(-1, 256)).reshape(bs, int(item_len/2), -1)
#            pred = self.fc(output.reshape(-1, 256)).reshape(bs, item_len, -1)[:, -self.opt.time_frame_length:, :] 
        
        return pred
    

# import torch
# model = Audio2Feature(output_length=91)
# x = torch.rand(16,90,45)
# print(model(x, output_length=90).shape)