from torch import nn
import torch


class Attention_lstm_model(nn.Module):
    def __init__(self, in_channel, input_size,  lstm_hdsize, num_lstm_layers, drop_out, num_heads, out_channel, middle_channel, num_step):
        super(Attention_lstm_model, self).__init__()

        self.in_channel = in_channel
        self.input_size = input_size
        self.lstm_hdsize = lstm_hdsize
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out
        self.num_heads = num_heads
        self.out_channel = out_channel
        self.num_step = num_step
        self.attention_emdsize = 2 * lstm_hdsize
        self.middle_channel = middle_channel

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, kernel_size=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, kernel_size=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.middle_channel),
            nn.Conv2d(self.middle_channel, self.out_channel, kernel_size=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.out_channel)

        )
        self.layer_norm_1 = nn.LayerNorm(self.input_size)
        self.lstm_layer = nn.LSTM(input_size=self.out_channel*self.input_size, hidden_size=self.lstm_hdsize,
                                  num_layers=self.num_lstm_layers, batch_first=True, bidirectional=True, dropout=self.drop_out)

        self.multi_attention = nn.MultiheadAttention(embed_dim=self.attention_emdsize, num_heads=self.num_heads,
                                                     batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(self.attention_emdsize)
        self.out_layer = nn.Sequential(
            nn.Linear(in_features=self.num_step * self.attention_emdsize, out_features=self.attention_emdsize),
            nn.Tanh(),
            nn.Linear(in_features=self.attention_emdsize, out_features=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
                
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        conv_out = self.conv_block(x)

        nrm_out_1 = self.layer_norm_1(conv_out + x)

        nrm_out_1 = nrm_out_1.reshape(nrm_out_1.shape[0], nrm_out_1.shape[2], nrm_out_1.shape[1]*nrm_out_1.shape[3])

        lstm_out, _ = self.lstm_layer(nrm_out_1)

        padding = torch.zeros(nrm_out_1.shape[0], nrm_out_1.shape[1], lstm_out.shape[2] - nrm_out_1.shape[2], device=nrm_out_1.device)

        nrm_out_1_padded = torch.cat((nrm_out_1, padding), dim=2)

        attention_out, _ = self.multi_attention(nrm_out_1_padded, lstm_out, lstm_out)

        norm_out_2 = self.layer_norm_2(attention_out + lstm_out)

        norm_out_2 = norm_out_2.reshape(norm_out_2.shape[0], norm_out_2.shape[1]*norm_out_2.shape[2])

        result = self.out_layer(norm_out_2).squeeze()

        return result



if __name__ == '__main__':
    input = torch.normal(0, 1, size=(64, 3, 10, 5))

    model = Attention_lstm_model(in_channel=3, input_size=5, lstm_hdsize=10, num_lstm_layers=2, drop_out=0.08, num_heads=1, out_channel=3, middle_channel=6,
                                 num_step=10)
    result = model(input)
    print(result.shape)



