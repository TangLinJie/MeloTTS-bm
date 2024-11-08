import os
import re
import json
import torch
import soundfile
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import warnings
import time

import utils
from split_utils import split_sentence
from sophon import sail

from engine import Engine


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

        # config_path = 
        if config_path is None:
            l = language.split('-')[0].upper()
            assert l in LANG_TO_HF_REPO_ID
            config_path = LANG_TO_HF_REPO_ID[l]
        hps = utils.get_hparams_from_file(config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # bmodel infer
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

        # option = onnxruntime.SessionOptions()
        # option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # option.intra_op_num_threads = 1
        # melo_onnx = "/home/yfchen/workspace/models/sherpa-onnx/vits-melo-tts-zh_en/model.onnx"
        # self.melo_session = onnxruntime.InferenceSession(melo_onnx, sess_options=option, providers=["CPUExecutionProvider"])
        melo_bmodel = "../bmodels/vits-melo-tts_1688_f16.bmodel" # "../bmodels/vits-melo-tts_1684x_f32.bmodel" # 
        # self.melo_infer = SGInfer(melo_bmodel, 1, [0])
        self.melo_infer = Engine(melo_bmodel, device_id=0, graph_id=0, mode=sail.IOMode.SYSIO)
        utils.init_jieba("开始", self.language)

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
        pipeline_st = time.time()
        st_time = time.time()
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet) # 将文本按指定语言分割成句子
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
        print("split cost time: ", time.time()-st_time)
        indx = 0
        for t in tx:
            st_time = time.time()
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t) # 将字符串 t 中的小写字母后面紧跟的大写字母之间插入一个空格
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
                print("preprocess cost time: ", time.time() - st_time)
                st_time = time.time()
                onnx_input = {
                    'x' : x_tst.detach().numpy(),
                    'x_lengths' : x_tst_lengths.detach().numpy(),
                    'tones' : tones.detach().numpy(),
                    'sid' : speakers.detach().numpy(),
                    'noise_scale' : torch.tensor([noise_scale], dtype=torch.float32).detach().numpy(),
                    'length_scale' : torch.tensor([1. / speed], dtype=torch.float32).detach().numpy(),
                    'noise_scale_w' : torch.tensor([noise_scale_w], dtype=torch.float32).detach().numpy(),
                }
                # print(f"x: {x_tst.shape}")

                # audio = self.melo_session.run(None, onnx_input)[0][0, 0]

                """
                bmodel_inputs = (
                    onnx_input['x'].astype(nptype(self.melo_infer.get_input_info()["x"]["dtype"])),
                    onnx_input['x_lengths'].astype(nptype(self.melo_infer.get_input_info()["x_lengths"]["dtype"])),
                    onnx_input['tones'].astype(nptype(self.melo_infer.get_input_info()["tones"]["dtype"])),
                    onnx_input['sid'].astype(nptype(self.melo_infer.get_input_info()["sid"]["dtype"])),
                    onnx_input['noise_scale'].astype(nptype(self.melo_infer.get_input_info()["noise_scale"]["dtype"])),
                    onnx_input['length_scale'].astype(nptype(self.melo_infer.get_input_info()["length_scale"]["dtype"])),
                    onnx_input['noise_scale_w'].astype(nptype(self.melo_infer.get_input_info()["noise_scale_w"]["dtype"])),
                )
                _ = self.melo_infer.put(*bmodel_inputs)
                _, bmodel_outputs, _ = self.melo_infer.get()
                """
                bmodel_inputs = (
                    onnx_input['x'],
                    onnx_input['x_lengths'],
                    onnx_input['tones'],
                    onnx_input['sid'],
                    onnx_input['noise_scale'],
                    onnx_input['length_scale'],
                    onnx_input['noise_scale_w'],
                )
                print("engine infer init cost time: ", time.time() - st_time)
                bmodel_outputs = self.melo_infer(bmodel_inputs)
                print("engine infer cost time: ", time.time() - st_time)
                bmodel_outputs[0] = torch.from_numpy(bmodel_outputs[0])
                audio = bmodel_outputs[0].squeeze((0, 1))

                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # import pdb; pdb.set_trace()
                # 
            audio_list.append(audio)
            # soundfile.write("test_output/{}.wav".format(indx), audio, self.hps.data.sampling_rate)
            indx += 1
            print("infer cost time: ", time.time() - st_time)
        st_time = time.time()
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
        print("pipeline cost: ", time.time() - pipeline_st)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
        print("write file cost time: ", time.time() - st_time)


if __name__ =="__main__":
    print("init start ...")
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
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
    # from melo.api import TTS
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    if language == 'EN':
        if not speaker: speaker = 'EN-Default'
        spkr = speaker_ids[speaker]
    else:
        spkr = speaker_ids[list(speaker_ids.keys())[0]]
    print("init end ...")
    model.tts_to_file(text, spkr, output_path, speed=speed)