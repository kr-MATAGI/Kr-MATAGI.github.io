from dataclasses import dataclass

@dataclass
class KT_TTS:
    id: str = '000000'
    sent: str = ""


class HangulJamo:
    def __init__(self):
        # @TODO : 영어에 대한 처리를 해야됨.
        # @TODO : 띄어쓰기 처리 추가 했고
        # @TODO : 숫자 처리
        # @TODO :

        ''' 'O' 는 기호같은거 처리하기 위해 '''
        self.initial = [
            ' ',
            'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ'
        ]
        self.medial = [
            '',
            'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ',
            'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
        ]
        self.final = [
            '',
            'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ',
            'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
            'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
        ]

        self.all_jamo = [
            '', ' ', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ',
            'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ',
            'ㅢ', 'ㅣ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ'
        ]

    def get_jamo_tok2ids(self):
        initial_tok2ids = {x: i for i, x in enumerate(self.initial)}
        medial_tok2ids = {x: i for i, x in enumerate(self.medial)}
        final_tok2ids = {x: i for i, x in enumerate(self.final)}

        return initial_tok2ids, medial_tok2ids, final_tok2ids

    def get_jamo_ids2tok(self):
        initial_ids2tok = {i: x for i, x in enumerate(self.initial)}
        medial_ids2tok = {i: x for i, x in enumerate(self.medial)}
        final_ids2tok = {i: x for i, x in enumerate(self.final)}

        return initial_ids2tok, medial_ids2tok, final_ids2tok

    def get_all_jamo_converter(self):
        all_jamo_ids2tok = {i: x for i, x in enumerate(self.all_jamo)}
        all_jamo_tok2ids = {x: i for i, x in enumerate(self.all_jamo)}

        return all_jamo_ids2tok, all_jamo_tok2ids

    def get_length(self):
        return len(self.all_jamo)

### main ###
if "__main__" == __name__:
    hangul_jamo = HangulJamo()
    all = ['']
    for a in hangul_jamo.initial:
        if a not in all:
            all.append(a)
    print(all)

    for b in hangul_jamo.medial:
        if b not in all:
            all.append(b)
    print(all)

    for c in hangul_jamo.final:
        if c not in all:
            all.append(c)
    print(all)
    print(len(all))
