# KcELECTRA: Korean comments ELECTRA

공개된 한국어 BERT는 대부분 한국어 위키, 뉴스 기사, 책 등 잘 정제된 데이터를 기반으로 학습한 모델입니다. 한편, 실제로 NSMC와 같은 댓글형 데이터셋은 정제되지 않았고 구어체 특징에 신조어가 많으며, 오탈자 등 공식적인 글쓰기에서 나타나지 않는 표현들이 빈번하게 등장합니다.

KcELECTRA는 위와 같은 특성의 데이터셋에 적용하기 위해, 네이버 뉴스에서 댓글과 대댓글을 수집해, 토크나이저와 ELECTRA모델을 처음부터 학습한 Pretrained ELECTRA 모델입니다.

기존 KcBERT 대비 데이터셋 증가 및 vocab 확장을 통해 상당한 수준으로 성능이 향상되었습니다.

KcELECTRA는 Huggingface의 Transformers 라이브러리를 통해 간편히 불러와 사용할 수 있습니다. (별도의 파일 다운로드가 필요하지 않습니다.)

```
💡 NOTE 💡 
General Corpus로 학습한 KoELECTRA가 보편적인 task에서는 성능이 더 잘 나올 가능성이 높습니다.
KcBERT/KcELECTRA는 User genrated, Noisy text에 대해서 보다 잘 동작하는 PLM입니다.
```

## KcELECTRA Performance

- Finetune 코드는 https://github.com/Beomi/KcBERT-finetune 에서 찾아보실 수 있습니다.
- 해당 Repo의 각 Checkpoint 폴더에서 Step별 세부 스코어를 확인하실 수 있습니다.

|                    | Size<br/>(용량) | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :----------------- | :-------------: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| **KcELECTRA-base** |      475M       |     **91.71**      |         86.90          |       74.80        |        81.65         |           82.65           |          **95.78**          |         70.60 / 90.11         |
| KcBERT-Base        |      417M       |       89.62        |         84.34          |       66.95        |        74.85         |           75.57           |            93.93            |         60.25 / 84.39         |
| KcBERT-Large       |      1.2G       |       90.68        |         85.53          |       70.15        |        76.99         |           77.49           |            94.06            |         62.16 / 86.64         |
| KoBERT             |      351M       |       89.63        |         86.11          |       80.65        |        79.00         |           79.64           |            93.93            |         52.81 / 80.27         |
| XLM-Roberta-Base   |      1.03G      |       89.49        |         86.26          |       82.95        |        79.92         |           79.09           |            93.53            |         64.70 / 88.94         |
| HanBERT            |      614M       |       90.16        |         87.31          |       82.40        |        80.89         |           83.33           |            94.19            |         78.74 / 92.02         |
| KoELECTRA-Base     |      423M       |       90.21        |         86.87          |       81.90        |        80.85         |           83.21           |            94.20            |         61.10 / 89.59         |
| KoELECTRA-Base-v2  |      423M       |       89.70        |         87.02          |       83.90        |        80.61         |           84.30           |            94.72            |         84.34 / 92.58         |
| KoELECTRA-Base-v3  |      423M       |       90.63        |       **88.11**        |     **84.45**      |      **82.24**       |         **85.53**         |            95.25            |       **84.83 / 93.45**       |
| DistilKoBERT       |      108M       |       88.41        |         84.13          |       62.55        |        70.55         |           73.21           |            92.48            |         54.12 / 77.80         |


\*HanBERT의 Size는 Bert Model과 Tokenizer DB를 합친 것입니다.

\***config의 세팅을 그대로 하여 돌린 결과이며, hyperparameter tuning을 추가적으로 할 시 더 좋은 성능이 나올 수 있습니다.**

## How to use

### Requirements

- `pytorch ~= 1.8.0`
- `transformers ~= 4.0.0`
- `emoji ~= 0.6.0`
- `soynlp ~= 0.0.493`

```python
from transformers import AutoTokenizer, AutoModel
  
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = AutoModel.from_pretrained("beomi/KcELECTRA-base")
```

### Pretrain & Finetune Colab 링크 모음 

#### Pretrain Data

- KcBERT학습에 사용한 데이터 + 이후 2021.03월 초까지 수집한 댓글
  - 약 17GB
  - 댓글-대댓글을 묶은 기반으로 Document 구성

#### Pretrain Code

- https://github.com/KLUE-benchmark/KLUE-ELECTRA Repo를 통한 Pretrain

#### Finetune Code

- https://github.com/Beomi/KcBERT-finetune Repo를 통한 Finetune 및 스코어 비교

#### Finetune Samples

- TBD

## Train Data & Preprocessing

### Raw Data

학습 데이터는 2019.01.01 ~ 2021.03.09 사이에 작성된 **댓글 많은 뉴스/혹은 전체 뉴스** 기사들의 **댓글과 대댓글**을 모두 수집한 데이터입니다.

데이터 사이즈는 텍스트만 추출시 **약 17.3GB이며, 1억8천만개 이상의 문장**으로 이뤄져 있습니다.

> KcBERT는 2019.01-2020.06의 텍스트로, 정제 후 약 9천만개 문장으로 학습을 진행했습니다.

### Preprocessing

PLM 학습을 위해서 전처리를 진행한 과정은 다음과 같습니다.

1. 한글 및 영어, 특수문자, 그리고 이모지(🥳)까지!

   정규표현식을 통해 한글, 영어, 특수문자를 포함해 Emoji까지 학습 대상에 포함했습니다.

   한편, 한글 범위를 `ㄱ-ㅎ가-힣` 으로 지정해 `ㄱ-힣` 내의 한자를 제외했습니다. 

2. 댓글 내 중복 문자열 축약

   `ㅋㅋㅋㅋㅋ`와 같이 중복된 글자를 `ㅋㅋ`와 같은 것으로 합쳤습니다.

3. Cased Model

   KcBERT는 영문에 대해서는 대소문자를 유지하는 Cased model입니다.

4. 글자 단위 10글자 이하 제거

   10글자 미만의 텍스트는 단일 단어로 이뤄진 경우가 많아 해당 부분을 제외했습니다.

5. 중복 제거

   중복적으로 쓰인 댓글을 제거하기 위해 완전히 일치하는 중복 댓글을 하나로 합쳤습니다.
   
6. `OOO` 제거

   네이버 댓글의 경우, 비속어는 자체 필터링을 통해 `OOO` 로 표시합니다. 이 부분을 공백으로 제거하였습니다.

아래 명령어로 pip로 설치한 뒤, 아래 clean함수로 클리닝을 하면 Downstream task에서 보다 성능이 좋아집니다. (`[UNK]` 감소)

```bash
pip install soynlp emoji
```

아래 `clean` 함수를 Text data에 사용해주세요.

```python
import re
import emoji
from soynlp.normalizer import repeat_normalize

emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x
```

> 💡 Finetune Score에서는 위 `clean` 함수를 적용하지 않았습니다.

### Cleaned Data

- KcBERT 외 추가 데이터는 정리 후 공개 예정입니다.


## Tokenizer Train

Tokenizer는 Huggingface의 [Tokenizers](https://github.com/huggingface/tokenizers) 라이브러리를 통해 학습을 진행했습니다.

그 중 `BertWordPieceTokenizer` 를 이용해 학습을 진행했고, Vocab Size는 `30000`으로 진행했습니다.

Tokenizer를 학습하는 것에는 전체 데이터를 통해 학습을 진행했고, 모델의 General Downstream task에 대응하기 위해 KoELECTRA에서 사용한 Vocab을 겹치지 않는 부분을 추가로 넣어주었습니다. (실제로 두 모델이 겹치는 부분은 약 5000토큰이었습니다.)

TPU `v3-8` 을 이용해 약 1주일의 학습을 진행했고, 현재 Huggingface에 공개된 모델은 848k step을 학습한 모델 weight가 업로드 되어있습니다.

모델 학습 Loss는 Step에 따라 초기 100-200k 사이에 급격히 Loss가 줄어들다 학습 종료까지도 지속적으로 loss가 감소하는 것을 볼 수 있습니다.

![KcELECTRA-base Pretrain Loss](https://cdn.jsdelivr.net/gh/beomi/blog-img@master/2021/04/07/image-20210407201231133.png)

## 인용표기/Citation

KcELECTRA를 인용하실 때는 아래 양식을 통해 인용해주세요.

```
@misc{lee2021,
  author = {Junbum Lee},
  title = {KcELECTRA: Korean comments ELECTRA},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Beomi/KcELECTRA}}
}
```

논문을 통한 사용 외에는 MIT 라이센스를 표기해주세요. ☺️

## Acknowledgement

KcELECTRA Model을 학습하는 GCP/TPU 환경은 [TFRC](https://www.tensorflow.org/tfrc?hl=ko) 프로그램의 지원을 받았습니다.

모델 학습 과정에서 많은 조언을 주신 [Monologg](https://github.com/monologg/) 님 감사합니다 :)

## Reference

### Github Repos

- [KcBERT by Beomi](https://github.com/Beomi/KcBERT)
- [BERT by Google](https://github.com/google-research/bert)
- [KoBERT by SKT](https://github.com/SKTBrain/KoBERT)
- [KoELECTRA by Monologg](https://github.com/monologg/KoELECTRA/)
- [Transformers by Huggingface](https://github.com/huggingface/transformers)
- [Tokenizers by Hugginface](https://github.com/huggingface/tokenizers)
- [ELECTRA train code by KLUE](https://github.com/KLUE-benchmark/KLUE-ELECTRA)

### Blogs

- [Monologg님의 KoELECTRA 학습기](https://monologg.kr/categories/NLP/ELECTRA/)
- [Colab에서 TPU로 BERT 처음부터 학습시키기 - Tensorflow/Google ver.](https://beomi.github.io/2020/02/26/Train-BERT-from-scratch-on-colab-TPU-Tensorflow-ver/)