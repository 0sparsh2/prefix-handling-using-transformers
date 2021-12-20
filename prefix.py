import nltk
from nltk.stem import PorterStemmer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

# From https://dictionary.cambridge.org/grammar/british-grammar/word-formation/prefixes
english_prefixes = {
"anti": "not",    # e.g. anti-goverment, anti-racist, anti-war
"auto": "automatically",    # e.g. autobiography, automobile
"de": "not",      # e.g. de-classify, decontaminate, demotivate
"dis": "not",     # e.g. disagree, displeasure, disqualify
#"down": "",    # e.g. downgrade, downhearted
"extra": "extremely",   # e.g. extraordinary, extraterrestrial
"hyper": "extreme",   # e.g. hyperactive, hypertension
#"il": "",     # e.g. illegal
"im": "not",     # e.g. impossible
"in": "not",     # e.g. insecure
#"ir": "",     # e.g. irregular
"inter": "amongst",  # e.g. interactive, international
"mega": "huge",   # e.g. megabyte, mega-deal, megaton
"mid": "middle of",    # e.g. midday, midnight, mid-October
"mis": "incorrect",    # e.g. misaligned, mislead, misspelt
"non": "not",    # e.g. non-payment, non-smoking
"over": "over",  # e.g. overcook, overcharge, overrate
#"out": "",    # e.g. outdo, out-perform, outrun
"post": "after",   # e.g. post-election, post-warn
"pre": "before",    # e.g. prehistoric, pre-war
"pro": "towards",    # e.g. pro-communist, pro-democracy
"re": "again",     # e.g. reconsider, redo, rewrite
"semi": "half",   # e.g. semicircle, semi-retired
#"sub": "",    # e.g. submarine, sub-Saharan
"super": "extreme",   # e.g. super-hero, supermodel
#"tele": "",    # e.g. television, telephathic
#"trans": "",   # e.g. transatlantic, transfer
#"ultra": "",   # e.g. ultra-compact, ultrasound
#"un": "",      # e.g. under-cook, underestimate
#"up": "",      # e.g. upgrade, uphill
#"re": " "      # e.g. reinstall, rereading
}

porter = PorterStemmer()

def stem_prefix(word, prefixes):
    original_word = word
    for prefix in sorted(prefixes, key=len, reverse=True):

        if word.startswith(prefix):
          len_pre = len(prefix)
          temp = word[len_pre:]

          sentence1 = word
          sentence2 = prefixes[prefix] + " " + temp 
          sentence3 = temp + " " +  prefixes[prefix]
          sentences = [sentence1, sentence2, sentence3]

          sentence_embeddings = model.encode(sentences)
          val = cosine_similarity(
                      [sentence_embeddings[0]],
                      sentence_embeddings[1:]
                  )
          print(sentences, prefix, val)

          max_val = 0.8
          result = word
          for i in range(len(val[0])):
            if val[0][i] >= max_val:
              max_val = val[0][i]
              result = sentences[i+1]

          print("Result:",result)

def porter_english_plus(word, prefixes=english_prefixes):
    return porter.stem(stem_prefix(word, prefixes))


input_word = input("Enter word to check")
stem_prefix(input_word, english_prefixes)