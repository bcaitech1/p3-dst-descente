# p3-dst-descente
p3-dst-descente created by GitHub Classroom

## 1. Create Data
- TRADE 의 input/data에 있는 train_dials, dev_dials, eval_dials를 다운로드 받습니다.

## 2. som-dst mapper
- train_dials와 dev_dials는 make_WOS_like_WOZ(file_path, is_eval=False)를 통해 som-dst 형식의 dials로 만들어줍니다.
- eval_dials는 make_WOS_like_WOZ(file_path, is_eval=True)를 통해 som-dst 형식의 dials로 만들어줍니다.
- /WizardOfSeoul/ 라는 디렉토리 생성해서 넣어줍니다.

## 3. ontology, slot_meta
- TRADE에 있는 ontology.json, slot_meta.json 을 /WizardOfSeoul/에 넣어줍니다.

## 3. train.py
- python3 train.py --save_dir outputs --op_code '4' 로 실행합니다.
