import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch

from tqdm import tqdm
from convec.inference import HubertFeatureReader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-v", "--vec", help="vec", dest="vec", required=True)
    
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)
    os.makedirs(args.vec, exist_ok=True)

    wavPath = args.wav
    vecPath = args.vec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert = HubertFeatureReader(os.path.join(
        "convec_pretrain", "checkpoint_best_legacy_500.pt"), device)

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{vecPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing vec {spks}'):
                file = file[:-4]
                hubert.get_feats(f"{wavPath}/{spks}/{file}.wav", f"{vecPath}/{spks}/{file}.vec")
