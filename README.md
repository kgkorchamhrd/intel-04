# 상공회의소 경기인력개발원 인텔교육 4기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kgkorchamhrd/intel-04.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 4기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: 조물조
<사용자의 얼굴을 바라보며 대화하는 LLM 탑재 로봇>
* Members
  | Name | Role |
  |----|----|
  | 이은서 | 팀장: 프로젝트 관리/시스템 통합/STM32 오디오 입출력 개발 |
  | 김민정 | 팀원: 라즈베리파이를 이용한 LLM 대화 로직 개발 |
  | 박명우 | 팀원: ESP32 Face Detection 및 통신 개발 |
  | 안진홍 | 팀원: STM32 및 하드웨어 개발/외관 제작 |

* Project Github : https://github.com/LES4975/my-little-chatbot
* 발표자료 : https://docs.google.com/presentation/d/1ke70n9bbn2_XCP1pXgg8Qt4nmO6muRPUy0RcIypigkQ/edit?usp=sharing

### Team : 아우모 레코 (Au-mo reco)
<객체와 카메라간 거리 측정 및 제스처 인식을 통한 자동 이동 요청 솔루션>
* Members
  | Name | Role |
  |----|----|
  | 문병일 | 팀장: 임베디드 하드웨어 설계 및 개발, 차량제어 로직 개발 |
  | 김경민 | 팀원: 사전학습모델, OpenVINO를 활용한 엣지 디바이스에서의 모델 최적화  |

* Project Github : https://github.com/david1597-embedded/aumo_reco_project
* 발표자료 : https://github.com/david1597-embedded/aumo_reco_project/blob/main/doc/aumo_reco_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C1.pptx

