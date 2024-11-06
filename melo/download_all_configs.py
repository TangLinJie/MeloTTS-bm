from huggingface_hub import hf_hub_download


LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}

def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")
        else:
            assert language in DOWNLOAD_CONFIG_URLS
            config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])
    print(language)
    print(config_path)
    return
    # return utils.get_hparams_from_file(config_path)


for l in LANG_TO_HF_REPO_ID.keys():
    load_or_download_config(l)