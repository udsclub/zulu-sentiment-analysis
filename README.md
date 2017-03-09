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
