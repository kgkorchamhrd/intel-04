import re
import unicodedata


def unicodeToAscii(s):
    s = re.sub(r'(?i)cc-by.*$', '', s)
    hangul_pattern = re.compile('[가-힣ㄱ-ㅎㅏ-ㅣ]')
    result = []
    for c in s:
        if hangul_pattern.match(c):
            result.append(c)
        else:
            for c_ in unicodedata.normalize('NFD', c):
                if unicodedata.category(c_) != 'Mn': 
                    result.append(c_)
    return ''.join(result)

def Norm_String(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ!?]+", r" ", s)
    return s.strip()

def read_language(L1, L2, reverse=False, verbose=False):
    print("read languages data...")
    pairs = []
    Encode_lang = []
    Decode_lang = []
    pf = open('%s2%s.txt' % (L1, L2), encoding='utf-8').read().strip().split('\n')
    for ll in pf:
        parts = ll.split('\t')
        if len(parts) >2:
            L1_lang = Norm_String(parts[0])
            L2_lang = Norm_String(parts[1])
            if reverse:
                pairs.append([L2_lang, L1_lang])
                Encode_lang.append(L2_lang)
                Decode_lang.append(L1_lang)
            else:
                pairs.append([L1_lang, L2_lang])
                Encode_lang.append(L1_lang)
                Decode_lang.append(L2_lang)
    if verbose: 
        print(pairs)
    return Encode_lang, Decode_lang, pairs