import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import fairseq

from fairseq.data.audio.audio_utils import get_features_or_waveform
# fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, device, layer=12):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.max_chunk = self.task.cfg.max_sample_size
        print(f"TASK CONFIG:\n{self.task.cfg}")

    def read_audio(self, path):
        wav = get_features_or_waveform(
            path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        return wav

    def get_feats(self, wavPath, vecPath):
        x = self.read_audio(wavPath)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                # print(start)
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        vec = torch.cat(feat, 1).squeeze(0)
        vec = vec.data.cpu().float().numpy()
        # print(vec.shape)  # [length, dim=768] hop=320
        np.save(vecPath, vec, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-v", "--vec", help="vec", dest="vec")
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)

    wavPath = args.wav
    vecPath = args.vec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    contentvec = HubertFeatureReader(os.path.join(
        "convec_pretrain", "checkpoint_best_legacy_500.pt"), device)
    contentvec.get_feats(wavPath, vecPath)
