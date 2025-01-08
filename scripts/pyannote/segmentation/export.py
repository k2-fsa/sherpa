#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import torch
import torch.nn.functional as F
from pyannote.audio import Model
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.core.utils.generators import pairwise
from torch import nn

"""
"linear":       {'hidden_size': 128, 'num_layers': 2}
"lstm":         {'hidden_size': 256, 'num_layers': 2, 'bidirectional': True, 'monolithic': True, 'dropout': 0.0, 'batch_first': True}
"num_channels": 1
"sample_rate":  16000
"sincnet":      {'stride': 10, 'sample_rate': 16000}
"""


class PyanNet(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.sincnet = SincNet(**m.hparams.sincnet)

        multi_layer_lstm = dict(m.hparams.lstm)
        del multi_layer_lstm["monolithic"]
        self.lstm = nn.LSTM(60, **multi_layer_lstm)

        lstm_out_features: int = m.hparams.lstm["hidden_size"] * (
            2 if m.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [m.hparams.linear["hidden_size"]] * m.hparams.linear["num_layers"]
                )
            ]
        )

        if m.hparams.linear["num_layers"] > 0:
            in_features = m.hparams.linear["hidden_size"]
        else:
            in_features = m.hparams.lstm["hidden_size"] * (
                2 if m.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, m.dimension)
        self.activation = m.default_activation()

    def forward(self, waveforms):
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)

        outputs, _ = self.lstm(torch.permute(outputs, (0, 2, 1)))

        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))


@torch.inference_mode()
def main():
    # You can download ./pytorch_model.bin from
    # https://hf-mirror.com/csukuangfj/pyannote-models/tree/main/segmentation-3.0
    # or from
    # https://huggingface.co/Revai/reverb-diarization-v1/tree/main
    pt_filename = "./pytorch_model.bin"
    model = Model.from_pretrained(pt_filename)
    wrapper = PyanNet(model)

    num_param1 = sum([p.numel() for p in model.parameters()])
    num_param2 = sum([p.numel() for p in wrapper.parameters()])

    assert num_param1 == num_param2, (num_param1, num_param2, model.hparams)
    print(f"Number of model parameters1: {num_param1}")
    print(f"Number of model parameters2: {num_param2}")

    model.eval()

    #  model.to_torchscript()  # won't work

    wrapper.eval()

    wrapper.load_state_dict(model.state_dict())

    x = torch.rand(1, 1, 10 * 16000)

    y1 = model(x)
    y2 = wrapper(x)

    assert y1.shape == y2.shape, (y1.shape, y2.shape)
    assert torch.allclose(y1, y2), (y1.sum(), y2.sum())

    m = torch.jit.script(wrapper)

    sample_rate = model.audio.sample_rate
    assert sample_rate == 16000, sample_rate

    window_size = int(model.specifications.duration) * 16000
    receptive_field_size = int(model.receptive_field.duration * 16000)
    receptive_field_shift = int(model.receptive_field.step * 16000)

    meta_data = {
        "num_speakers": str(len(model.specifications.classes)),
        "powerset_max_classes": str(model.specifications.powerset_max_classes),
        "num_classes": str(model.dimension),
        "sample_rate": str(sample_rate),
        "window_size": str(window_size),
        "receptive_field_size": str(receptive_field_size),
        "receptive_field_shift": str(receptive_field_shift),
        "model_type": "pyannote-segmentation-3.0",
        "version": "1",
        "maintainer": "k2-fsa",
    }

    m.save("model.pt", _extra_files=meta_data)
    print(meta_data)


if __name__ == "__main__":
    torch.manual_seed(20240108)
    main()
