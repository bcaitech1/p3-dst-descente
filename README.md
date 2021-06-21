# KLUE-DST Benchmark for TRADE & SOM-DST

<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122718082-74730b00-d2a7-11eb-90ef-d2ab562b172b.png" width=450px></p>

2021년 5월 21일 공개된 KLUE 데이터셋 중, `wos`(wizard of seoul) 데이터셋에 대한 벤치마크 스코어를 도출 및 기록하였습니다.
사용한 Pretrained Language Model은 KLUE에서 공개한 `KLUE/BERT-base`, `KLUE/RoBERTa-base`, `KLUE/RoBERTa-large` 입니다. (추후 `KLUE/RoBERTa-small` 및 `koelectra-base` 추가 예정입니다.) 

<p align='center'><img src="https://user-images.githubusercontent.com/37925813/122719518-3d9df480-d2a9-11eb-933c-28feb7df49d5.png" width=800px></p>

현재 KLUE Leader Board가 활성화되지 않은 관계로, 해당 벤치마크 스코어는 `dev set`에 대한 스코어임을 알려드립니다. (추후 Test Set 공개 시 스코어를 수정할 예정입니다.)


## Benchmark Score for TRADE & SOM-DST
| DST Model            | Pretrained Language Model | Epoch | Batch Size | Learning Rate           | JGA    | Slot F1 |
| -------------------- | ------------------------- | ----- | ---------- | ----------------------- | ------ | ------- |
| TRADE<br>(improved)  | KLUE/BERT-base            | 30    | 4          | 3e-5                    | 0.6401 | 0.9521  |
| TRADE<br/>(improved) | KLUE/RoBERTa-base         | 30    | 4          | 1e-5                    | 0.6311 | 0.9486  |
| TRADE<br/>(improved) | KLUE/RoBERTa-large        | 30    | 4          | 1e-5                    | 0.7119 | 0.9625  |
| SOM-DST              | KLUE/BERT-base            | 50    | 16         | enc: 4e-5<br>dec: 1e-4  | 0.6620 | 0.9552  |
| SOM-DST              | KLUE/RoBERTa-base         | 50    | 16         | enc: 4e-5<br/>dec: 1e-4 | 0.6301 | 0.9503  |
| SOM-DST              | KLUE/RoBERTa-large        | 30    | 4          | enc: 1e-5<br/>dec: 1e-4 | 0.6548 | 0.9566  |



