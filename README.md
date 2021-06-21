# KLUE-DST Benchmark for TRADE & SOM-DST

![Generic badge](https://img.shields.io/badge/unstable-v1.0.0-orange.svg)
<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122718082-74730b00-d2a7-11eb-90ef-d2ab562b172b.png" width=300px></p>

2021년 5월 21일 공개된 KLUE 데이터셋 중, `wos`(wizard of seoul) 데이터셋에 대한 벤치마크 스코어를 도출 및 기록하였습니다.
사용한 Pretrained Language Model은 KLUE에서 공개한 `KLUE/BERT-base`, `KLUE/RoBERTa-base`, `KLUE/RoBERTa-large` 입니다. (추후 `KLUE/RoBERTa-small` 및 `KoELECTRA-base` 추가 예정입니다.) 
<br>

<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122719518-3d9df480-d2a9-11eb-933c-28feb7df49d5.png" width=800px></p>
<br>

현재 KLUE Leader Board가 활성화되지 않은 관계로, 해당 벤치마크 스코어는 `dev set`에 대한 스코어를 기준으로 작성되었습니다.<br> 
(추후 Test Set 공개 시 스코어를 수정할 예정입니다.)

<br>

## 0. Requirements
- python 3.7
- torch==1.8.0
- transformers==4.6.1

<br>

## 1. Benchmark Score for TRADE & SOM-DST (on dev set)

| DST Model            | Pretrained Language Model | Epoch | Batch Size | Learning Rate           | JGA    | Slot F1 |
| -------------------- | ------------------------- | ----- | ---------- | ----------------------- | ------ | ------- |
| TRADE<br>(improved)  | KLUE/BERT-base            | 30    | 4          | 3e-5                    | 0.6401 | 0.9521  |
| TRADE<br/>(improved) | KLUE/RoBERTa-base         | 30    | 4          | 1e-5                    | 0.6311 | 0.9486  |
| TRADE<br/>(improved) | KLUE/RoBERTa-large        | 30    | 4          | 1e-5                    | 0.7119 | 0.9625  |
| SOM-DST              | KLUE/BERT-base            | 50    | 16         | enc: 4e-5<br>dec: 1e-4  | 0.6620 | 0.9552  |
| SOM-DST              | KLUE/RoBERTa-base         | 50    | 16         | enc: 4e-5<br/>dec: 1e-4 | 0.6301 | 0.9503  |
| SOM-DST              | KLUE/RoBERTa-large        | 30    | 4          | enc: 1e-5<br/>dec: 1e-4 | 0.6548 | 0.9566  |

<br>

## 2. Benchmark Score of TRADE model with epochs restricted(5, 10 epochs, on dev set)
KLUE 벤치마크 스코어 도출 환경과 유사한 환경에서 실험을 진행하기 위해, 우리의 TRADE 모델에 대해 epoch를 `5` 와 `10`으로 제한했을 때의 성능을 비교해보았습니다.
(회색 음영 처리된 부분이 기존 `wos` 데이터셋에 대한 KLUE의 벤치마크 스코어입니다. 정확한 실험환경을 알 수 없어 기존 모델의 Epoch = `5` 로 가정했습니다.)
<br>

<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122720132-09770380-d2aa-11eb-8c71-0ee3f5b4aab0.png" width=600px></p>
<br>

## 3. Benchmark Score for TRADE & SOM-DST (on `edited` dev set)
<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122720724-c701f680-d2aa-11eb-8bc9-586441da331f.png" width=600px></p>
<br>
wos-v1 dev set에 대한 ontology 및 dialogues EDA를 수행하면서 총 4가지 종류의 문제점을 파악했습니다.<br><br>

```
1) Mixed Ontologies
  - '서울중앙성원남산골 한옥 마을', '이럴순없는대호스텔 제이제이'와 같이, 두 개의 ontology가 분리되지 않고 합쳐져 있는 오류입니다.
2) Wrong Ontologies
  - '동대문사문화공원역'(정식명칭 : 동대문역사문화공원역), '신도역'(정식명칭 : 신도림역) 등 잘못 표기된 ontology. 
  - '홍대역'/'홍익대학교역'(정식명칭: 홍대입구역) 등 같은 지하철 역명에 대한 다중표기 문제가 존재합니다.
3) Simple Mistakes 
  - ① 오타 문제 - ex. '위잉키친'(옳은 표기 : '위잉치킨')
  - ② 띄어쓰기 문제 - ex. '강남이네 게스트하우스'(옳은 표기: '강남이네 게스트 하우스')
  - ③ Missing State - ex. "같은 지역에 한식당을 찾아달라"는 user의 발화가 있었지만 '식당-종류-한식당' 이라는 라벨이 없는 경우
  - ④ Wrong State - ex. "적당한 가격대의 에어비엔비를 알아봐달라"는 user의 발화에 '숙소-가격대-저렴'이라는 state를 정답으로 표기.
4) Location/Time Problems 
  - ① 장소 유추 문제 : "그럼 숙소가 있는 역에서 광화문역 가려면 중간에 환승 해야하나요?” 에서, user가 언급한 '숙소가 있는 역'을 유추할 수 없는 문제 
  - ② 시간 유추 문제 : "좋아요. 곧 출발 할 거니까 9시 50분으로 2명 예약해주세요.” 에서, user가 언급한 '곧' 이 무슨 요일인지 유추할 수 없는 문제
```

<br>
<br>
이에 대한 개선방안을 논의해본 결과, 아래와 같은 개선방안을 제시해보고자 하였습니다.<br><br>
<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122722016-39bfa180-d2ac-11eb-89fa-e7273331b115.png" width=800px></p>
<br>
이 중, 가장 많은 비중을 차지하고 있는 3번(simple mistakes) 문제점을 일부 수정하여 edited dev set을 만들었고, 이에 대한 벤치마크 스코어를 아래의 표와 같이 새로 도출해보았습니다.<br><br>
<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122722614-e568f180-d2ac-11eb-9e1a-badd9c2042b7.png" width=1000px></p><br>

모델에 따라 JGA 기준으로 1.5%p ~ 2.36%p 정도의 스코어 향상을 나타냈습니다.<br>
추후 동일 결과를 재현할 수 있도록 코드 업데이트 예정입니다. 👀