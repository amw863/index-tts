from indextts.infer import IndexTTS

if __name__ == "__main__":
    prompt_wav="./data/voice/sample_prompt.wav"
    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
#     text = """《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，
# 莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，
# 2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。
# 影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，
# 讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。
# """.replace("\n", "")
    # 读取文本
    text = open("./data/text/sample_text.txt", "r", encoding="utf-8").read().replace("\n", "")
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
