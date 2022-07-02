from entity.format.base import BaseEntityFormatter


class EntityFormatter(BaseEntityFormatter):
    """ Форматирование типов сущностей в тексте.

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

        Предлагается не экранировать следующие сущности:
        ```
        AGE, AWARD, CRIME, DISTRICT, FAMILY, IDEOLOGY, PENALTY, RELIGION, PROFESSION
        ```
    """

    # Ключи
    AGE = "AGE"
    AWARD = "AWARD"
    CITY = "CITY"
    COUNTRY = "COUNTRY"
    CRIME = "CRIME"
    DATE = "DATE"
    DISEASE = "DISEASE"
    DISTRICT = "DISTRICT"
    EVENT = "EVENT"
    FACILITY = "FACILITY"
    FAMILY = "FAMILY"
    IDEOLOGY = "IDEOLOGY"
    LANGUAGE = "LANGUAGE"
    LAW = "LAW"
    LOCATION = "LOCATION"
    MONEY = "MONEY"
    NATIONALITY = "NATIONALITY"
    NUMBER = "NUMBER"
    ORDINAL = "ORDINAL"
    ORGANIZATION = "ORGANIZATION"
    PENALTY = "PENALTY"
    PERCENT = "PERCENT"
    PERSON = "PERSON"
    PRODUCT = "PRODUCT"
    PROFESSION = "PROFESSION"
    RELIGION = "RELIGION"
    STATE_OR_PROVINCE = "STATE_OR_PROVINCE"
    TIME = "TIME"
    WORK_OF_ART = "WORK_OF_ART"

    __types_fmt = {
        AGE: "возраст",
        AWARD: "награда",
        CITY: "город",
        COUNTRY: "страна",
        CRIME: "преступление",
        DATE: "дата",
        DISEASE: "болезнь",
        DISTRICT: "район",
        EVENT: "событие",
        FACILITY: "сооружение",
        FAMILY: "семья",
        IDEOLOGY: "идеология",
        LANGUAGE: "язык",
        LAW: "закон",
        LOCATION: "локация",
        MONEY: "средства",
        NATIONALITY: "национальность",
        NUMBER: "количество",
        ORDINAL: "номер",
        ORGANIZATION: "организация",
        PENALTY: "штраф",
        PERCENT: "процент",
        PERSON: "личность",
        PRODUCT: "продукт",
        PROFESSION: "профессия",
        RELIGION: "религия",
        STATE_OR_PROVINCE: "штат",
        TIME: "время",
        WORK_OF_ART: "исскуство"
    }

    # Можно полагаться на BERT-ontonotes, в котором поддерживаются следующие типы:
    # http://docs.deeppavlov.ai/en/master/features/models/ner.html#named-entity-recognition-ner
    __supported_list = [
        # AGE
        # AWARD
        # CRIME
        # DISTRICT
        # FAMILY,
        # IDEOLOGY,
        # PENALTY,
        # RELIGION,
        # PROFESSION,
        NATIONALITY,            # NORP
        DATE,
        STATE_OR_PROVINCE,      # GPE-like
        LAW,
        LANGUAGE,
        LOCATION,
        MONEY,
        ORGANIZATION,
        PERCENT,
        PERSON,
        PRODUCT,
        EVENT,
        CITY,                   # LOCATION-like
        COUNTRY,                # LOCATION-like
        ORDINAL,
        NUMBER,
        FACILITY,
        TIME,
        WORK_OF_ART
    ]

    __supported_set = set(__supported_list)

    def format(self, entity_value, entity_type, is_target):
        assert(isinstance(entity_value, str))
        assert(isinstance(entity_type, str))
        return EntityFormatter.__types_fmt[entity_type] \
            if entity_type in EntityFormatter.__supported_set else entity_value
