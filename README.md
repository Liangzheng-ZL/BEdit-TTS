# BEDIT-TTS: TEXT-BASED SPEECH EDITING SYSTEM WITH BIDIRECTIONAL TRANSFORMERS

In our paper, we proposed BEdit-TTS: Text-Based Speech Editing System with Bidirectional Transformers. We provide our [speech samples](https://anonymous.4open.science/w/bedit_web-9718/) and code as open source in this repository.

## Set up
The system is built on [ESPnet](https://github.com/espnet/espnet). 
Before running the model, please install ESPnet.
This model requires Python 3.7+ and Pytorch 1.10+. 
Other packages are listed in requirements.yaml.

## Data
The kaldi-style data (text, utt2spk, spk2utt, wav.scp) has been provided in data directory.
To obtain duration information, you can use the [kaldi tool](https://kaldi-asr.org/) to train the GMM-HMM model to achieve forced alignment.
Dictionary and all texts of HiFiTTS can be downloaded from [drive.google](https://drive.google.com/file/d/1IwK60nhXQw3fac3r3qIkpRHpk14b1YHP/view?usp=sharing).

To extract feature:
```bash
bash run.sh --stage 1 --stop_stage 1
```
To apply CMVN:
```bash
bash run.sh --stage 1 --stop_stage 1
```
To prepar dictionary and json data:
```bash
bash run.sh --stage 1 --stop_stage 1
```
To update json data:
```bash
bash run.sh --stage 1 --stop_stage 1
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



