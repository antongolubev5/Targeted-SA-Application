class BaseEntityFormatter(object):
    """ Интерфейс форматирования сущностей
    """

    def format(self, entity_value, entity_type, is_target):
        """ entity_value: значение сущности
            entity_type: BRAT тип сущности
            is_target: флаг, указывающий на то, являтся ли сущность таргетом
        """
        raise NotImplementedError()
