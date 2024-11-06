import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import warnings

import utils
import commons
from models import SynthesizerTrn
from split_utils import split_sentence
from download_utils import load_or_download_model


LANG_TO_HF_REPO_ID = {
    'EN': 'configs/EN_config.json',
    'EN_V2': 'configs/EN_V2_config.json',
    'EN_NEWEST': 'configs/EN_NEWEST_config.json',
    'FR': 'configs/FR_config.json',
    'JP': 'configs/JP_config.json',
    'ES': 'configs/ES_config.json',
    'ZH': 'configs/ZH_config.json',
    'KR': 'configs/KR_config.json',
}

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        if config_path is None:
            l = language.split('-')[0].upper()
            assert l in LANG_TO_HF_REPO_ID
            config_path = LANG_TO_HF_REPO_ID[l]
        hps = utils.get_hparams_from_file(config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        # bmodel infer
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)


if __name__ =="__main__":
    text = "2023年至今，Deep Learning 正在飞速发展，在N-L-P领域，以Chat G-P-T为代表的L-L-M模型对长文本的理解有了巨大的提升，使得A-I具有了深度思考的能力。A-I-G-C领域也迎来了爆发，诞生了处理Text-to-Image、Image-to-Image任务的Stable-Difusion、S-D-X-L、ControlNet模型，以及文生视频的Sora模型。"
    output_path = "./test_output/zh.wav"
    file = False # help="Text is a file"
    language = 'ZH' # help='Language, defaults to English', type=click.Choice(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], case_sensitive=False))
    speaker = 'EN-Default' # help='Speaker ID, only for English, leave empty for default, ignored if not English. If English, defaults to "EN-Default"', type=click.Choice(['EN-Default', 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU']))
    speed = 1.0 # help='Speed, defaults to 1.0', type=float)
    device = 'auto' # help='Device, defaults to auto')
    if file:
        if not os.path.exists(text):
            raise FileNotFoundError(f'Trying to load text from file due to --file/-f flag, but file not found. Remove the --file/-f flag to pass a string.')
        else:
            with open(text) as f:
                text = f.read().strip()
    if text == '':
        raise ValueError('You entered empty text or the file you passed was empty.')
    language = language.upper()
    if language == '': language = 'EN'
    if speaker == '': speaker = None
    if (not language == 'EN') and speaker:
        warnings.warn('You specified a speaker but the language is English.')
    from melo.api import TTS
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    if language == 'EN':
        if not speaker: speaker = 'EN-Default'
        spkr = speaker_ids[speaker]
    else:
        spkr = speaker_ids[list(speaker_ids.keys())[0]]
    model.tts_to_file(text, spkr, output_path, speed=speed)