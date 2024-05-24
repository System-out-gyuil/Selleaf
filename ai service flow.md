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


메인페이지에서 상단에 노하우가 인기순으로 뜸
로그인시 회원의 인사이트를 기반으로 10개가 노출
5개 3개 2개
노하우(사용자가 자신의 경험을 공유하기 위해 글을 올리는 장소, 커뮤니티 기능)

사용자에게 처음으로 노출되는 메인페이지이므로 사용자 유입을 유도할 수 있어야함
유도를 위해서는 사용자의 관심도가 높은 컨텐츠를 노출했을 때 가능성이 높아짐
때문에 비회원일때 회원일때를 나눠서 훈련하는 AI 시스템으로

비회원일때는 대중적인 컨텐츠를 제공하고 (인기순)
회원일때는 회원의 데이터를 기반으로 나열하여 맞춤형 서비스를 제공하고자 함


기획 배경

1. 사용자의 재방문을 유도하는 지속성있는 컨텐츠 노출이 필요함 (플랫폼 성장)

2. 참여율을 높이기 위한 맞춤형 데이터 기반 서비스 제공


기획 의도

1. 사용자의 참여율 상승 (유저 증가로 인해 수익적인 면에서 좋음)

2. 탐색 피로도 감소효과 , 탐색 편의성 증가

3. 고객 데이터를 기반으로 하여 개인화 효과를 극대화

우리 사이트는 고객 맞춤형 서비스를 통해 개인화 효과를 극대화합니다. 이를 통해 고객은 자신이 특별한 대우를 받고 있다고 느끼게 되며, 이는 고객 만족도와 충성도를 크게 향상시킵니다. 예를 들어, 고객의 선호도와 과거 구매 이력을 기반으로 추천 상품을 제공하고, 개인 맞춤형 프로모션을 제안함으로써 고객 경험을 최적화합니다
