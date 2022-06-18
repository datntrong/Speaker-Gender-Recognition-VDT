import torch.nn as nn
import wavencoder

class SpectralCNNLSTM(nn.Module):
    def __init__(self, lstm_h):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(40, int(lstm_h / 2), 5),
            nn.ReLU(),
            nn.BatchNorm1d(int(lstm_h / 2)),
            nn.Conv1d(int(lstm_h / 2), int(lstm_h / 2), 5),
            nn.ReLU(),
            nn.BatchNorm1d(int(lstm_h / 2)),
        )
        self.lstm = nn.LSTM(int(lstm_h / 2), int(lstm_h / 2), batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(int(lstm_h / 2), int(lstm_h / 2))

        self.gender_classifier = nn.Sequential(
            nn.Linear(int(lstm_h / 2), int(lstm_h / 2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(lstm_h / 2), 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(1)
        x = self.encoder(x)
        output, (hidden, _) = self.lstm(x.transpose(1, 2))
        attn_output = self.attention(output)

        gender = self.gender_classifier(attn_output)
        return gender


class SpectralLSTM(nn.Module):
    def __init__(self, lstm_h):
        super().__init__()

        self.lstm = nn.LSTM(128, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h)

        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_h, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(1)
        output, (hidden, _) = self.lstm(x.transpose(1, 2))
        attn_output = self.attention(output)

        gender = self.gender_classifier(attn_output)
        return gender


class Wav2VecLSTM(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h)
        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        output, (hidden, _) = self.lstm(x.transpose(1, 2))
        attn_output = self.attention(output)
        gender = self.gender_classifier(attn_output)
        return gender