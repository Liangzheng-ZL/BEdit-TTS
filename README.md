# BEDIT-TTS: TEXT-BASED SPEECH EDITING SYSTEM WITH BIDIRECTIONAL TRANSFORMERS

In our paper, we proposed BEdit-TTS: Text-Based Speech Editing System with Bidirectional Transformers. We provide our [speech samples](https://anonymous.4open.science/w/bedit-web-7468/index.html) and code as open source in this repository.

The model code is at ```espnet/nets/pytorch_backend/e2e_tts_bedit.py```

## Set up
The system is built on [ESPnet](https://github.com/espnet/espnet). 
Before running the model, please install ESPnet.
This model requires Python 3.7+ and Pytorch 1.10+. 
Other packages are listed in requirements.yaml.

## Data
To obtain duration information, you can use the [kaldi tool](https://kaldi-asr.org/) to train the GMM-HMM model to achieve forced alignment.

To prepare the data of BEdit-TTS:
```bash
bash run.sh --stage 0 --stop_stage 0
bash pre_bedit_data.sh
```
To apply CMVN:
```bash
bash run.sh --stage 1 --stop_stage 1
```
To prepar dictionary and json data:
```bash
bash run.sh --stage 2 --stop_stage 2
```
To update json data:
```bash
bash run.sh --stage 3 --stop_stage 3
```

## Run
To train the model:
```bash
bash run.sh --stage 4 --stop_stage 4
```
To generate spectrum:
```bash
bash run.sh --stage 5 --stop_stage 5
```
The waveform can be synthesized by a pre-trained HiFiGAN.



