from entity.format.base import BaseEntityFormatter


class TargetOnlyEntityFormatter(BaseEntityFormatter):
    """ Простая реализация форматирования сущностей.
        Сущности записываются как есть, за исключением таргетированной.
    """

    def format(self, entity_value, entity_type, is_target):
        assert(isinstance(entity_value, str))
        assert(isinstance(is_target, bool))

        return "location - 1" if is_target else entity_value
