import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def trans_to_en(texts: list):
    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                 download_method=DownloadMethod.REUSE_RESOURCES, use_gpu=True)

    text_en_l = []
    in_docs = [stanza.Document([], text = d) for d in texts]
    out_docs = nlp_stanza(in_docs)
    for doc in out_docs:
        doc_text = doc.text
        print(doc.lang)
        if doc.lang == "en":
            text_en_l = text_en_l + [doc_text]
        else:
            lang = doc.lang
            if lang == "fro":  # fro = Old French
                lang = "fr"
            elif lang == "la":  # latin
                lang = "it"
            elif lang == "nn":  # Norwegian (Nynorsk)
                lang = "no"
            elif lang == "kmr":  # Kurmanji
                lang = "tr"
            elif lang == "mt":
                lang = "en"

            case = 2

            if case == 1:
                text_en = t2t_pipe(doc_text, forced_bos_token_id=t2t_pipe.tokenizer.get_lang_id(lang='en'))
                text_en = text_en[0]['generated_text']
            elif case == 2:
                tokenizer.src_lang = lang
                encoded_hi = tokenizer(doc_text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                text_en = text_en[0]
            else:
                text_en = doc_text

            text_en_l = text_en_l + [text_en]

            print(doc_text)
            print(text_en)

    return text_en_l
