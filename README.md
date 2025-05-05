안녕하세요, SSAFY AI Challenge 대회 참가팀 ‘싸피라띠노 랄라벨로’의 팀장 문선웅입니다.

저는 RTX 3060 8GB 그래픽카드를 사용하여 로컬 환경에서 모델을 학습시켰습니다.
이번 대회에서는 팀원이 학습 도중 중단한 YOLOv8l 모델의 `best.pt`를 이어받아 추가 학습을 진행했습니다.

SSAFY AI Challenge에서 사용한 학습 코드는 `ai.py`입니다.

## 데이터 준비

* 대회에서 제공한 데이터를 다운로드한 후 압축을 해제하세요.
* 압축을 해제하면 `train`, `valid`, `test` 폴더가 생성되며, 이 폴더들을 `ai.py` 파일과 동일한 디렉토리에 위치시켜야 합니다.

![디렉토리 구조](/photos/directory_hierarchy.png)

※ 압축 해제는 시간이 다소 소요되므로 잠시 기다려 주세요.

## 환경 설정

1. **CUDA Toolkit**: 구글에서 'CUDA Toolkit'을 검색하여 설치하세요.
2. **cuDNN**: 구글에서 'cuDNN'을 검색하여 설치하세요.
3. **PyTorch**:

   * [PyTorch 공식 사이트](https://pytorch.org/)에 접속하여 본인의 환경에 맞는 설치 명령어를 확인하세요.
   * 예시 명령어 (CUDA 12.1 기준):

     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   * ※ 버전에 따라 URL이나 명령어가 다를 수 있습니다. 반드시 사이트에서 확인 바랍니다.

![PyTorch 설치 가이드](/photos/pytorch_site.png)

## 학습 실행

`ai.py` 파일을 실행하면 다음과 같은 폴더 구조가 생성됩니다:

```
finetune/
└── weights/
    ├── best.pt
    └── last.pt
```

* 매 에포크마다 `weights` 폴더 안의 `best.pt`와 `last.pt`가 갱신됩니다.
* 학습이 모두 종료되면, 가장 성능이 좋은 모델인 `best.pt`를 `ai.py`와 동일한 디렉토리에 복사하거나 덮어씌워 저장하세요.

이상이 SSAFY AI Challenge 코드 실행 및 학습에 필요한 전체 가이드입니다.
감사합니다.
