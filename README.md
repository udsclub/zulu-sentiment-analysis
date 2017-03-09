# Team Zulu - Linear models

## 1 step
Первой моделью для теста мы выбрали LogisticRegression (далее LR) и использовали CountVectorizer (далее CV).
Препроцессинг: stop words, удаление не английских комментариев, удаление        цифровых токенов.
Валидация на RT: 0.76 

## 2 step
Тестировали RidgeClassifier и SGDClassifier и CV векторизацию.

**Algorithm**|**Train**|**Test**|**Validation accuracy**|**Test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
RidgeClassifier|RT|IMDB|0,7682|0,7815
RidgeClassifier|IMDB|RT|0,8832|0,7116
SGDClassifier|IMDB|RT|0,863|0,7076
SGDClassifier|RT|IMDB|0,7825|0,8105
