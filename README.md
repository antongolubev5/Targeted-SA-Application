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

Сущности фиксируются как есть, экранируется только та сущность, для которой предсказывается класс
[[code]](entity/format/target_only.py)
