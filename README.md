# 자소제한 후처리 기법을 이용한 한국어 발음 변환

### 실행하기전에...

1. 아래의 경로에서 Data를 다운로드 해주세요.

```
KT 데이터 비공개
```

1-2. 필요하시다면 아래의 경로에서 Model을 다운로드 해주세요.

&nbsp;&nbsp;&nbsp;&nbsp; - 다운로드 후 test_model에서 압축을 해제하시면 됩니다.

```
...
```

2. 실행에 필요한 requirements를 설치해주세요.

&nbsp;&nbsp;&nbsp;&nbsp; - python3.8 이상을 권장합니다.

```
pip install -r requirements.txt
```

### 훈련 및 테스트

1. config 파일을 설정해주세요.

&nbsp;&nbsp;&nbsp;&nbsp; - 경로 : config/kocharelectra_config.json</p>

```json
{
  "ckpt_dir": "ko-char-electra-encoder-decoder", -> 모델이 저장될 가장 상위 폴더의 바로 아래 하위 폴더
  "train_npy": "./data/data_busan/kor/npy/train.npy",
  "dev_npy": "./data/data_busan/kor/npy/dev.npy",
  "test_npy": "./data/data_busan/kor/npy/test.npy",
  "evaluate_test_during_training": true,
  "eval_all_checkpoints": true,
  "save_optimizer": false,
  "do_train": false, -> 훈련을 수행할지
  "do_eval": true, -> 테스트를 수행할지
  "max_seq_len": 256,
  "num_train_epochs": 10,
  "weight_decay": 0.0,
  "gradient_accumulation_steps": 1,
  "adam_epsilon": 1e-8,
  "warmup_proportion": 0,
  "max_grad_norm": 1.0,
  "model_type": "electra-base",
  "model_name_or_path": "monologg/kocharelectra-base-discriminator",
  "output_dir": "./test_model/ckpt-ar-end2end", -> 모델이 저장될 가장 상위 폴더 명
  "seed": 42,
  "train_batch_size": 32,
  "eval_batch_size": 128,
  "logging_steps": 858, -> 모델 훈련시 몇 step 마다 검증 테스트를 수행할지
  "save_steps": 858, -> 모델 훈련시 몇 step 마다 저장을 할지
  "learning_rate": 5e-5
}
```

2. 아래 명령어를 실행해주세요.

```
python run_g2p.py
```

### 임의의 문장을 테스트 하고 싶은 경우

```
python ar_test.py --input=안녕하세요 \
--ckpt_path=test_model/ckpt-ar-end2end\ko-char-electra-encoder-decoder\checkpoint-17150 \
--config_path=config/kocharelectra_config.json \
--decoder_vocab_path=data/vocab/decoder_vocab/pron_eumjeol_vocab.json \
--jaso_dict_path=data/vocab/post_process/jaso_filter.json
```
