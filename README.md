# Team Zulu - Linear models

## 1 step
Первой моделью для теста мы выбрали `LogisticRegression` (далее `LR`) и использовали `CountVectorizer` (далее `CV`).
Препроцессинг: stop words, удаление не английских комментариев, удаление цифровых токенов.
Валидация на RT: 0.76 

## 2 step
Тестировали `RidgeClassifier` и `SGDClassifier` и `CV` векторизацию.

**Algorithm**|**Train**|**Test**|**Validation accuracy**|**Test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
RidgeClassifier|RT|IMDB|0,7682|0,7815
RidgeClassifier|IMDB|RT|0,8832|0,7116
SGDClassifier|IMDB|RT|0,863|0,7076
SGDClassifier|RT|IMDB|0,7825|0,8105

## 3 step
Тестирование `LR` с гипермараметрами (подбирали вручную) и `CV` векторизацией.
Гиперпараметры: `penalty='l2', cv=10, coring='roc_auc',random_state=777,max_iter=10000, fit_intercept=True, solver='newton-cg' ,tol=20`

**Algorithm**|**Train**|**Test**|**Validation accuracy**|**Test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
LogisticRegressionCV|RT|IMDB|0,7945|0,8437
LogisticRegressionCV|IMDB|RT|0,8965|0,7286

## 4 step
Тестирование `PassiveAggressiveClassifier` (далее `PAC`) и `CV` векторизацией.
Гиперпараметры: `C=0.01, fit_intercept = False, shuffle = True, n_iter = 10`

**Algorithm**|**Train**|**Test**|**Validation accuracy**|**Test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
PassiveAggressiveClassifier|RT|IMDB|0,7949|0,8526
PassiveAggressiveClassifier|IMDB|RT|0,8658|0,7190

## 5 step
Тестирование `MultinomialNB`, `svm.LinearSVC`,  `BernoulliNB` и `CV` векторизацией.Также, в `MultinomialNB` для RT - IMDB использовался `Tf-Idf`, а для IMDB - RT `Ngrams(1,3)`. В каждом из вариантов эти модификации давали лучший вариант. 

**Algorithm**|**Train**|**Test**|**Validation accuracy**|**Test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
MultinomialNB|RT|IMDB|0.7753|0.8393
MultinomialNB|IMDB|RT|0.8983|0.7729
svm.LinearSVC|RT|IMDB|0.7647|0.8156
svm.LinearSVC|IMDB|RT|0.8674|0.6889
BernoulliNB|RT|IMDB|0.7821|0.7269
BernoulliNB|IMDB|RT|0.8566|0.7329

## 6 step
Улучшение `PAC`, с подбром лучших параметров (циклом), а также использованием `Ngrams(1,3)`  и стоп-словами. Веторизация `CV`
Гиперпараметры: `(C=0.001, fit_intercept = False, shuffle =   False, n_iter = 91, n_jobs = -1)`
Стоп-слова: `['a','by','does', 'was', 'were', 'the','i', 'of', 'and', 'to', 'is']`

**Algorithm**|**Train**|**Test**|**Validation accuracy**|**Test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
PassiveAggressiveClassifier|RT|IMDB|0.8098|0.8770
PassiveAggressiveClassifier|IMDB|RT|0.9088|0.7420

Дальше мы еще играли со стоп-словами и `Ngrams`, но изменения там были `+-0,0005` 

## 7 step
Заменили `CV` на `word2vec`, для `PAC`.
Гиперпараметры: `(C=0.001, fit_intercept = False, shuffle =   False, n_iter = 91, n_jobs = -1)`

**Algorithm**|**Train**|**Validation accuracy**
:-----:|:-----:|:-----:
PassiveAggressiveClassifier|RT|0.7345
PassiveAggressiveClassifier|RT+IMDB|0.7339

В связи с ухудшением валидации, решили его не использовать.

## 8 step
Поиск оптимального количества слов с конца, для `PAC` с векторизацией `CV`. 
Предварительно мы разбили датасет RT и IMDB на трейн/тест (80/20). Затем смешали их и на смешанном тренировались, обрезая определенное количество слов.
Тестирование проводили на всех трех.
Гиперамараметры: `(C=0.001, fit_intercept = False, shuffle =   False, n_iter = 91, n_jobs = -1)`

**Count of words**|**Accuracy**| | 
:-----:|:-----:|:-----:|:-----:
 |RT|IMDB|RT+IMDB
10|0.803333008479|0.8873|0.830843326125
15|0.808644381639|0.898|0.837920188716
18|0.810983334958|0.9001|0.840180853155
20|0.810739693987|0.9031|0.840999934473
30|0.810788422181|0.9015|0.840508485682
40|0.810496053016|0.8999|0.839787694122

Так как в районе 20 тест на всех трех выборках ведет себя наилучшим образом, мы детализирован.

**Count of words**|**Accuracy**| | 
:-----:|:-----:|:-----:|:-----:
 |RT|IMDB|RT+IMDB
18|0.810983334958|0.9001|0.840180853155
19|0.810203683851|0.9016|0.840148089902
20|0.810739693987|0.9031|0.840999934473
21|0.811811714258|0.9034|0.841819015792
22|0.812396452587|0.9034|0.842212174825
23|0.812493908976|0.901|0.841491383265
24|0.811421888705|0.9024|0.841229277243

Более гармоничные результаты тест показывает на 22 словах.

## 9 step
Для модели описанной выше, произвели подбор лучше стоп-слов. Поочередно добавляя их в список и замеряя валидацию. При улучшении слово оставалось в списке, при ухудшении из него удалялось.
`STOPWORDS = ['of', 'im', 'that', 'for', 'film', 'are', 'its', 'the']`

**RT**|**IMDB**|**RT+IMDB**
:-----:|:-----:|:-----:
0.813565929247|0.9022|0.842605333858

## 10 step
C помощью `GridSearchCV` нашел самые оптимальные параметры для логрегрессии:
`LogisticRegression(solver='liblinear', C=1.0, fit_intercept=False)`,      
`CountVectorizer(binary=True,ngram_range=(1,4))`.
Ниже приведена таблица, которая показывает параметры которые использовались для поиска оптимальных

**LogisticRegression&rsquo;sParameters**|**Values**
:-----:|:-----:
C|0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0
solver|‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’
fit_intercept|True, False

**Count of words**|**RT Acc**|**IMDB Acc**|**RT+IMDB Acc**|**StopWords**
:-----:|:-----:|:-----:|:-----:|:-----:
22 from the end|0.8071|0.8959|0.8362|-
20 from the beginning and 20 from the end|0.8094|0.9012|0.8395|-
22 from the end|0.8114|0.9014|0.8409|+
20 from the beginning and 20 from the end|0.8082|0.9027|0.8392|+
22 from the beginning and 22 from the end|0.8088|0.9033|0.8398|+

## Step 11
C помощью `GridSearchCV` нашли оптимальные параметры для `Support Vector Machines`:
`LinearSVC(random_state=42,fit_intercept=False,C=0.1)`,
`CountVectorizer(binary = True,ngram_range=(1,3),stop_words=STOPWORDS)`,      
Результаты чуть хуже, чем для `PassiveAgressiveClassifier`:

 |**Accuracy RT**|**Accuracy IMDB**|**Accuracy RT+IMDB**
:-----:|:-----:|:-----:|:-----:
LinearSVC|0.811519345093|0.9036|0.841687962781
PassiveAggressive|0.812396452587|0.9034|0.842212174825

## Step 12
Testing custom features. The text was cut do last 22 words before Feature extraction
Result: a very small improvement with one of the features - "positive smiles":

**Custom Features**|**RT+IMDB accuracy**
:-----:|:-----:
no|0.842212174825
sentence|0.842212174825
question mark count|0.841819015792
exclamation mark count|0.841982832056
uppercase|0.842244938077
uppercase + rating|0.83975493087
pos smiles (no uppercase)|0.84227770133
uppercase + neg smiles|0.842244938077
uppercase + contrast|0.84155690977
uppercase + polarity last|0.838149531485
uppercase + subjectivity last|0.838673743529
uppercase + polarity all|0.842244938077
uppercase + purity|0.839296245331

 |**Accuracy RT**|**Accuracy IMDB**|**Accuracy RT+IMDB**
:-----:|:-----:|:-----:|:-----:
Positive Smiles|0.812445180782|0.9035|0.84227770133
No custom features|0.812396452587|0.9034|0.842212174825

## Step 13
Basic Text Preprocessing again

**Basic preprocessing**|**RT+IMDB**
:-----:|:-----:
no|0.84227770133
drop duplicates|0.837596707819
drop non-english reviews|0.839396951624
lowercase=False|0.838346111002
