## Разметка предложений в документах 

Используется модель `ru_core_news_sm`:
```python
nlp = spacy.load('ru_core_news_sm')
```

## Разбиение данных
**Фиксированное**: [[code]](run_split_fixed.py)
* Выполняется по соотношению TRAIN = 70%, TEST = 30%
```
Среднее число предложений в TRAIN 17.7636
Среднее число предложений в TEST 17.7542
```

**Кросс-валидационное**: [[code]](run_split_k_fold_cv.py)
* 5-Fold Cross-Validataion;
* документы добавляются поочередно и в тот fold, в котором суммарно меньше всего предложений.
```
Статистика разбиений: [1397, 1395, 1394, 1397, 1397]
```
    
## Представление именованных сущностей в тексте

* Вместо значений именованных сущностей **используются их типы**.
* Для типов применяется русскоязычное форматирование [[code]](entity_fmt.py)

Из типов были отобраны такие, которые заведомо могут быть распознаны NER.
(В целях потенциальной возможности применения моделей на сырых текстах, где нет разметки сущностей)
За основу взята модель BERT-onto-notes. 
[[link]](http://docs.deeppavlov.ai/en/master/features/models/ner.html#named-entity-recognition-ner)

```
PERSON          People including fictional
NORP            Nationalities or religious or political groups
FACILITY        Buildings, airports, highways, bridges, etc.
ORGANIZATION    Companies, agencies, institutions, etc.
GPE             ountries, cities, states
LOCATION        Non-GPE locations, mountain ranges, bodies of water
PRODUCT         Vehicles, weapons, foods, etc. (Not services)
EVENT           Named hurricanes, battles, wars, sports events, etc.
WORK OF ART     Titles of books, songs, etc.
LAW             Named documents made into laws
LANGUAGE        Any named language
DATE            Absolute or relative dates or periods
TIME            Times smaller than a day
PERCENT         Percentage (including “%”)
MONEY           Monetary values, including unit
QUANTITY        Measurements, as of weight or distance
ORDINAL         “first”, “second”
CARDINAL        Numerals that do not fall under another type
```