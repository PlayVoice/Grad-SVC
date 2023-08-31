<div align="center">
<h1> Grad-SVC based on Grad-TTS from HUAWEI Noah's Ark Lab </h1>

This project is named as [Grad-SVC](), or [GVC]() for short. Its core technology is diffusion, but so different from other diffusion based SVC models. Codes are adapted from `Grad-TTS` and `so-vits-svc-5.0`. So the features from `so-vits-svc-5.0` will be used in this project. And [Diff-VC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC) is a follow-up of [Grad-TTS](), [Diffusion-Based Any-to-Any Voice Conversion](https://arxiv.org/abs/2109.13821)

The project will be completed in the coming months ~~~

[Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337)

![grad_tts](./assets/grad_tts.jpg)

![diff_vc](./assets/diff_vc.jpg)

![grad_svc](./assets/grad_svc.jpg)

The framework of grad-svc

![CoMoSpeech](./assets/CoMoSpeech.png)

To Be integrated

</div>

## Features
1. Multi-speaker based on speaker encoder

2. No speaker leaky based on `GRL`

3. No electronic sound

## Setup Environment
1. Install project dependencies

    ```shell
    pip install -r requirements.txt
    ```

2. Download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put `best_model.pth.tar`  into `speaker_pretrain/`.

3. Download [hubert_soft model](https://github.com/bshall/hubert/releases/tag/v0.1)，put `hubert-soft-0d54a1f4.pt` into `hubert_pretrain/`.

4. Download pretrained [nsf_bigvgan_pretrain_32K.pth](https://github.com/PlayVoice/NSF-BigVGAN/releases/augment), and put it into `bigvgan_pretrain/`.

5. Download pretrain model [gvc.pretrain.pth](https://github.com/PlayVoice/Grad-SVC/releases/tag/20230829), and put it into `grad_pretrain/`.
    > python gvc_inference.py --config configs/base.yaml --model ./grad_pretrain/gvc.pretrain.pth --spk ./configs/singers/singer0001.npy --wave test.wav
    
    For this pretrain model, `temperature` is set `temperature=1.015` in `gvc_inference.py` to get good result.
   
## Dataset preparation
Put the dataset into the `data_raw` directory following the structure below.
```
data_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
    ├───000001.wav
    ├───...
    └───000xxx.wav
```

## Data preprocessing
After preprocessing you will get an output with following structure.
```
data_gvc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── mel
│    └── speaker0
│    │      ├── 000001.mel.pt
│    │      └── 000xxx.mel.pt
│    └── speaker1
│           ├── 000001.mel.pt
│           └── 000xxx.mel.pt
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── hubert
│    └── speaker0
│    │      ├── 000001.vec.npy
│    │      └── 000xxx.vec.npy
│    └── speaker1
│           ├── 000001.vec.npy
│           └── 000xxx.vec.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
    ├── speaker0.spk.npy
    └── speaker1.spk.npy
```

1.  Re-sampling
    - Generate audio with a sampling rate of 16000Hz in `./data_gvc/waves-16k` 
    ```
    python prepare/preprocess_a.py -w ./data_raw -o ./data_gvc/waves-16k -s 16000
    ```
    - Generate audio with a sampling rate of 32000Hz in `./data_gvc/waves-32k`
    ```
    python prepare/preprocess_a.py -w ./data_raw -o ./data_gvc/waves-32k -s 32000
    ```
2. Use 16K audio to extract pitch
    ```
    python prepare/preprocess_f0.py -w data_gvc/waves-16k/ -p data_gvc/pitch
    ```
3. use 32k audio to extract mel
    ```
    python prepare/preprocess_spec.py -w data_gvc/waves-32k/ -s data_gvc/mel
    ``` 
4. Use 16K audio to extract hubert
    ```
    python prepare/preprocess_hubert.py -w data_gvc/waves-16k/ -v data_gvc/hubert
    ```
5. Use 16k audio to extract timbre code
    ```
    python prepare/preprocess_speaker.py data_gvc/waves-16k/ data_gvc/speaker
    ```
6. Extract the average value of the timbre code for inference
    ```
    python prepare/preprocess_speaker_ave.py data_gvc/speaker/ data_gvc/singer
    ``` 
8. Use 32k audio to generate training index
    ```
    python prepare/preprocess_train.py
    ```
9. Training file debugging
    ```
    python prepare/preprocess_zzz.py
    ```

## Train
1. Start training
   ```
   python gvc_trainer.py
   ``` 
2. Resume training
   ```
   python gvc_trainer.py -p logs/grad_svc/grad_svc_***.pth
   ```
3. Log visualization
   ```
   tensorboard --logdir logs/
   ```

## Loss
![grad_svc_loss](./assets/grad_svc_loss.jpg)

![grad_svc_mel](./assets/grad_svc_mel.jpg)

## Inference

1. Export inference model
   ```
   python gvc_export.py --checkpoint_path logs/grad_svc/grad_svc_***.pt
   ```

2. Inference
    - Convert wave to mel
        > python gvc_inference.py --model gvc.pth --spk ./data_gvc/singer/your_singer.spk.npy --wave test.wav --shift 0
    - Convert mel to wave
        ```
        python gvc_inference_wave.py --mel gvc_out.mel.pt --pit gvc_tmp.pit.csv
        ```

2. Inference step by step
    - Extract hubert content vector
        ```
        python hubert/inference.py -w test.wav -v test.vec.npy
        ```
    - Extract pitch to the csv text format
        ```
        python pitch/inference.py -w test.wav -p test.csv
        ```
    - Convert hubert & pitch to mel
        > python gvc_inference.py --model gvc.pth --spk ./data_gvc/singer/your_singer.spk.npy --wave test.wav --vec test.vec.npy --pit test.csv --shift 0
    - Convert mel to wave
        ```
        python gvc_inference_wave.py --mel gvc_out.mel.pt --pit test.csv
        ```

## Code sources and references

https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS [paper](https://arxiv.org/abs/2105.06337)

https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC [paper](https://arxiv.org/abs/2109.13821)

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/shivammehta25/Diff-TTSG [paper](https://arxiv.org/abs/2306.09417)

https://github.com/zhenye234/CoMoSpeech [paper](https://arxiv.org/abs/2305.06908)

https://github.com/lmnt-com/wavegrad [paper](https://arxiv.org/pdf/2009.00713.pdf)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/mindslab-ai/univnet [paper](https://arxiv.org/abs/2106.07889)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/bshall/soft-vc [paper](https://arxiv.org/abs/2111.02392)

https://github.com/mozilla/TTS

https://github.com/maxrmorrison/torchcrepe

## QQ Grop
<div align="center">

![X·SING-QQ](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/dfbb4740-447e-40f0-9193-c974c4b6ceb0)

</div>
