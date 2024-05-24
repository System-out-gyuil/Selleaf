<br>
<br>
<br>

![logo-lettering](https://github.com/DianaKang0123/selleaf/assets/156397873/b5f4c8cd-6d88-4965-9336-ad89f151ba52)

# 머신러닝 웹 적용 프로젝트

<br>

## 나이브베이즈 분류: 게시물 카테고리 추천 시스템

### **👍 목차**

1. 개요
2. 데이터 수집
3. 데이터 전처리 및 모델 훈련
4. 추천 알고리즘의 흐름
5. 서비스 기대효과
6. 트러블 슈팅
7. 느낀점

<br>

---

<br>

### **1️⃣ 개요**

<br>

셀리프의 커뮤니티에는 카테고리 서비스가 있습니다. 이 **카테고리**는 게시물을 그룹화하여 사용자가 원하는 게시글을 모아서 볼 수 있도록 도와주는 기능 입니다.

**나이브베이즈를 통한 카테고리 추천 시스템**은 노하우에 우선적으로 적용되며, **노하우게시글 상세보기** 단계에서 회원이 들어가본 노하우 게시글의 제목과 내용을 텍스트로 결합하여 최근에 본 게시글을 통해 카테고리를 추천해서 메인페이지에서 추천된 노하우 게시글들을 볼 수 있습니다.

<br>

---

<br>

### **2️⃣ 데이터 수집**

<br>

#### 🚩 모델의 사전 훈련용 데이터를 모으는 과정은 다음과 같습니다.

    https://www.gardening.news/

- 위 사이트의 게시글을 크롤링을 통하여 홈페이지 성격에 맞는 식물에 관련 된 글의 제목, 내용을 발췌하여 사용
- 사이트의 게시글별 카테고리를 따로 추가해주어서 사용

<br>

---

<br>

### **3️⃣ 데이터 전처리 및 모델 훈련**

#### 🚩 데이터 전처리.

- 우선 CountVectorizer 사용을 위해 데이터의 제목과 내용을 이어서 붙혀주었습니다.  
   ex) 제목 : 안녕하세요, 내용 : 반갑습니다 ▶️ feature : 안녕하세요 반갑습니다.

- 타겟피쳐인 카테고리가 문자열 형식으로 되어있어서 모델 학습을 위해  
  LabelEncoder를 통해 0부터 시작되는 연속된 숫자로 변환해주었습니다.

  > 0 - 꽃  
  > 1 - 농촌  
  > 2 - 원예  
  > 3 - 정원

<br>
<hr>

#### CountVectorizer

- 노하우 게시글의 제목과 내용을 벡터화하여 텍스트 데이터를 단어의 빈도수로 반환합니다.

#### MultinomialNB

- 위에서 CountVectorizer를 통해 반환받은 단어의 빈도수를 이용하여 문자의 빈도, 즉 확률을 구합니다.

<br>

- 파이프라인을 통해 CountVectorizer와 MultinomialNB를 사용하여 훈련하였을 때

  ![knowhow_eva01](https://github.com/System-out-gyuil/django_with_ai/assets/120631088/e7af7801-31ff-4485-a680-5c15362abc29)

  정확도: 0.6844, 정밀도: 0.6620, 재현율: 0.6352, F1: 0.6309

- 결과가 그렇게 나쁘지 않았으나 위의 시각화를 통해 확인할 수 있듯이 타겟의 비중이 맞지않아  
  정답이 한쪽에 쏠려있는 모습을 볼 수 있습니다.

- 따라서 타겟의 비중을 맞추기 위하여 언더샘플링을 진행하였습니다.

- 언더샘플링 후  
  ![knowhow_eva02](https://github.com/System-out-gyuil/django_with_ai/assets/120631088/014dceb7-2b54-4779-9ccf-8f4c190356a2)
  정확도: 0.5580, 정밀도: 0.6526, 재현율: 0.5796, F1: 0.5387

- 수치가 조금 떨어지긴 하였으나 언더샘플링을 진행하기 전엔 임의의 값을 넣고 predict를 하였을 경우  
   결과가 자꾸 한쪽으로만 나타나서 언더샘플링을 사용한채로 진행하였습니다.

<br>

---

<br>

### **4️⃣ 추천 알고리즘의 흐름**

<br>

#### 추천 알고리즘의 흐름은 다음과 같습니다.

1. 회원가입 시 사전훈련모델을 베이스로 개인의 모델이 생성됩니다.  
   <img width="300" alt="knowhow_pkl01" src="https://github.com/System-out-gyuil/django_with_ai/assets/120631088/f0433941-9cb0-4884-b147-987b7202743e">

   > 개인 모델의 pkl파일이 생성된 모습.

  <details>  
    <summary>모델생성 코드 보기</summary>

        # 사전 훈련모델을 불러와서 개인 모델 생성
        knowhow_ai_models = {}
        knowhow_ai_models[f'knowhow_ai{member.id}'] = joblib.load(os.path.join(Path(__file__).resolve().parent, '../main/ai/knowhow_ai.pkl'))

        joblib.dump(knowhow_ai_models[f'knowhow_ai{member.id}'], f'../main/ai/knowhow_ai{member.id}.pkl')

        # 저장할 파일의 경로를 지정
        file_path = os.path.join(Path(__file__).resolve().parent, f'../main/ai/knowhow_ai{member.id}.pkl')
        directory = os.path.dirname(file_path)

        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 모델을 지정된 경로에 저장
        joblib.dump(knowhow_ai_models[f'knowhow_ai{member.id}'], file_path)

        # member 테이블의 member_knowhow_ad_model 컬럼에 경로 저장
        member_model = Member.objects.get(id=member.id)
        member_model.member_knowhow_ai_model = f'main/ai/knowhow_ai{member.id}.pkl'
        member_model.save(update_fields=['member_knowhow_ai_model'])

  </details>  
<br>
2. 노하우 상세보기 페이지에 들어갈때 누가 어떤 게시글을 보았는지에 대한 정보를 저장하고, 위에서 생성된 개인 모델을 추가학습 시킵니다.
  <details>  
    <summary>정보 저장 및 추가학습 코드 보기</summary>

        # 어느 회원이 어떤 게시글을 보았는지
        KnowhowView.objects.create(knowhow_id=knowhow.id, member_id=session_member_id)

        # 개인 모델 불러오기
        knowhow_model = joblib.load(
            os.path.join(Path(__file__).resolve().parent, f'../main/ai/knowhow_ai{session_member_id}.pkl')
        )

        knowhow_title = Knowhow.objects.filter(id=knowhow.id).values('knowhow_title')
        knowhow_content = Knowhow.objects.filter(id=knowhow.id).values('knowhow_content')
        knowhow_category = KnowhowCategory.objects.filter(knowhow_id=knowhow.id).values('category_name')

        knowhow_feature = knowhow_title[0]['knowhow_title'] + " " + knowhow_content[0]['knowhow_content']
        target_dict = {
            '꽃': 0,
            '농촌': 1,
            '원예': 2,
            '정원': 3
        }

        knowhow_target = target_dict[knowhow_category[0].get('category_name')]

        # 모델 학습
        transformd_features = knowhow_model.named_steps['count_vectorizer'].transform([knowhow_feature])
        knowhow_model.named_steps['nb'].partial_fit(transformd_features, [knowhow_target])

        # 저장할 파일의 경로를 지정
        file_path = os.path.join(Path(__file__).resolve().parent, f'../main/ai/knowhow_ai{session_member_id}.pkl')
        directory = os.path.dirname(file_path)

        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 모델을 지정된 경로에 저장
        joblib.dump(knowhow_model, file_path)

  </details>
  <br>
3. 위에서 훈련된 모델과 회원이 본 게시글의 정보를 통해 카테고리를 추천합니다.
    <details>  
       <summary>카테고리 추천 코드 보기</summary>

          # 회원이 본 노하우 게시글이 3개 이상일 때
          if member_object and KnowhowView.objects.filter(member_id=member_object.id).count() >= 3:

          # 훈련된 개인 모델을 불러옴
          knowhow_model = joblib.load(os.path.join(Path(__file__).resolve().parent, f'ai/knowhow_ai{member_object.id}.pkl'))

          # 최근에 본 세개의 노하우 게시물
          knowhow_id = KnowhowView.objects.filter(member_id=member_object.id).order_by('-id')[:3].values('knowhow_id')

          # 가져온 노하우 게시물의 갯수만큼 반복
          knowhows = [0] * len(knowhow_id)
          probas = [0] * len(knowhow_id)
          for i in range(len(knowhow_id)):

              # 노하우 게시물의 제목과 내용
              knowhows[i] = Knowhow.objects.filter(id=knowhow_id[i].get('knowhow_id')).values('knowhow_title', 'knowhow_content')

              # 제목과 내용을 연결
              knowhows[i] = (knowhows[i][0]['knowhow_title']) + (knowhows[i][0]['knowhow_content'])

              # 위에서 연결한 제목과 내용을 개인 모델에 predict_proba를 사용하여 각각의 확률(우선순위)을 구해줌
              probas[i] = knowhow_model.predict_proba([knowhows[i]])

          total_proba = [0] * len(probas[0][0])
          for i in range(len(total_proba)):

              # 각 노하우 게시물별 확률을 합산
              total_proba[i] = (probas[0][0][i] + probas[1][0][i] + probas[2][0][i])

          print('total_proba', total_proba)

          categories = ['꽃', '농촌', '원예', '정원']
          knowhows = []
          np_total_proba = np.array(total_proba)
          # arg_sort를 통해 인덱싱 및 높은 순 정렬
          argsorted_indices = np_total_proba.argsort()[::-1]

          # 순위별 불러온 게시글 갯수
          amounts = [5, 3, 2, 0]

          for i in range(4):

              # 정렬된 순으로 카테고리를 넣어줌
              category_name = categories[argsorted_indices[i]]
              category_amount = amounts[i]

              # 카테고리를 통해 카테고리별 최신순으로 불러옴, 각 카테고리당 amounts의 갯수대로 불러옴
              knowhows += list(Knowhow.objects.filter(knowhowcategory__category_name=category_name).order_by('-id')[:category_amount]\
                              .annotate(member_profile=F('member__memberprofile__file_url'),
                                        member_name=F('member__member_name')) \
                              .values('member_profile', 'member_name', 'id', 'knowhow_title'))

</details>

<hr>

### **5️⃣ 서비스 기대효과**

1. 탐색 편의성 증가

- 게시글의 다양한 카테고리 중 유저가 자주보거나 관심이 있는 분야의 게시물을 메인페이지에 노출시켜서  
  게시물 목록에 들어가서 스크롤을 내리며 찾아보지 않아도 추천된 게시물을 통해 편한 탐색이 가능해집니다.

2. 플랫폼 성장

- 개인화된 추천을 통해 사용자의 만족도를 높여 이탈률을 낮춰주며, 사용자가 사이트에서 긍정적인 경험을 하고, 주변에 공유함으로써 새로운 사용자를 유입할 수 있습니다.

3. 효율성 증대

- 사용자들이 자주 이용하는 카테고리등을 직접 분석하지 않아도 자동으로 분석하고 사용자에게 맞는 게시물을 나타내기에  
  운영의 효율성을 증대시킬 수 있습니다.

4. 수익 증대

- 유입이 증가하고, 이탈이 적어짐에 따라 전체 유저의 수가 늘어남으로 인해서 자연스럽게 수익이 증가하고,  
  더욱 발전시켜 유료 서비스나 프리미엄 컨텐츠등을 추가하거나 전환하는등 긍정적인 비젼이 생깁니다.

<hr>

### **6️⃣트러블 슈팅**

### - 경로 에러
#### 1. 문제 발생
Jupyter notebook에서 pkl파일로 내보낸 모델을 Pycharm에서 불러오려 할 때 joblib의 load를 통해 불러왔었는데  
분명 Jupyter notebook에선 잘 불러와져서 사용이 가능하였었는데 파이참에서 똑같이 사용하였을 경우  
`FileNotFoundError: [Errno 2] No such file or directory: 'knowhow_ai.pkl'`  
라는 에러가 나타났습니다.

#### 2. 원인 확인
해당 에러는 설정된 경로에 해당 파일 혹은 폴더가 없다는 에러였습니다.  
따라서 정확한 경로를 설정해주어야 한다고 생각하였습니다.   

#### 3. 문제 해결
`joblib.load(os.path.join(Path(**file**).resolve().parent, 'ai/knowhow_ai19.pkl'))` os.path.join을 통해 경로를 문자로 연결하고  
Path를 사용하여 파일 시스템에서 경로를 사용할 수 있게 해준 후 resolve()를 통해 절대경로로 변환하고 parent로 부모 폴더로부터 경로를 입력하여 경로를 간소화하고, 이 후 원하는 파일을 불러와 사용해주었습니다.

<hr>

### - 모델 학습 에러(파이프라인, 추가 학습)
#### 1. 문제 발생
개인 모델을 추가학습하는 과정에서  
`knowhow_model.fit(transformd_features, [knowhow_target])` 으로 불러온 모델에 feature와 target을 넣고 fit하였는데  
파이프라인을 불러오지 못하는 문제가 생겼었습니다.  

#### 2. 원인 확인
해당 에러는 모델에 사용된 파이프라인에서 각각의 객체를 따로 훈련해주어야 하는데, 모델 자체에 그냥 fit을 하여 발생한 오류였습니다.
따라서 파이프라인을 하나씩 불러와서 사용해주어야 한다고 생각하였습니다.

#### 3. 문제 해결
`transformd_features = knowhow_model.named_steps['count_vectorizer'].transform([knowhow_feature])`
`knowhow_model.named_steps['nb'].fit(transformd_features, [knowhow_target])`  
named_steps를 사용하여 사전훈련모델에 파이프라인으로 사용했던 객체를 각각 가져와서 먼저 feature를 count_vectorizer를 통해  
변환시킨 후 MultinomialNB를 사용하여 추가학습을 진행하였습니다.

#### 4. 새로운 문제 발생
위에에서 추가학습된 모델을 통해 예측하는 값에 이상이 생겼었습니다.
메인화면에서 개인에게 추천된 게시글을 보여주어야하는데 계속 똑같은 게시글만 나타나는 문제가 있었습니다.

#### 5. 원인 확인
원래 predict하여 나오는 값이  
`[2.362991167550716, 0.004116118195193652, 0.6169707099866734, 0.015922004267416696]` 이런식으로  
카테고리 하나당 하나씩 확률이 나와야하는데 추가학습 이후 확률이 `[3.0]`으로 고정되어버린다는걸 확인하였습니다.

#### 6. 문제 해결
그냥 fit을 하면 해당 시점에서의 학습된 값 하나로 학습하고 덮어씌워지기 때문에 MultinomialNB모델에서 지원하는 partial_fit을 사용하여 해결하였습니다.  
`transformd_features = knowhow_model.named_steps['count_vectorizer'].transform([knowhow_feature])`
`knowhow_model.named_steps['nb'].partial_fit(transformd_features, [knowhow_target])`

<hr>

### **7️⃣ 느낀점**

- 머신러닝으로 프로젝트를 진행하며 항상 어느정도 정제된 데이터를 가지고 분류 혹은 회귀를 진행 하였는데, 직접 데이터를 수집하고 모델을 훈련시키는 과정에서부터
  어려움이 있었는데 그래도 직접 데이터를 넣고 예측하고 결과를 확인하고 다시 데이터를 수집하고 그런 과정을 겪으며, 데이터가 부족하거나 문제가 있을때 직접 해결할 수 있다는 점이 굉장히 흥미로웠습니다.

- 웹 프로젝트에 모델을 적용시켜 각 회원마다 회워의 데이터를 기반으로 알고리즘이 적용되어
  내가 이때까지 보던 많은 웹사이트들에서 나에게 맞는 게시글등이 나타나는것에 대하여 이해할 수 있게 되어서 정말 신기한 경험이었습니다.

- 사전 훈련모델을 만들때는 확률이 50% ~ 60% 정도로 낮은 결과였지만 그럼에도 내가 만든 웹사이트에서 실제 데이터로 예측을 하였을 때 생각보다 너무 잘 나와서 머신러닝이 얼마나 대단한지 느꼈습니다.


