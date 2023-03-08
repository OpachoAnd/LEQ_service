from sqlalchemy.orm import sessionmaker

from db.config import ENGINE
from db.models import Base


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DBManager(metaclass=SingletonMeta):
    connection = None

    @classmethod
    def connect(cls):
        if cls.connection is None:
            cls.connection = sessionmaker(bind=ENGINE)()
        return cls.connection


class DBCommands:
    def __init__(self):
        self.pool = DBManager.connect()

    @staticmethod
    def create_tables():
        """
        Создает таблицы в БД
        """
        Base.metadata.create_all(ENGINE)


if __name__ == '__main__':
    # Создание текущих таблиц при начале работы с базой данных
    DBCommands.create_tables()
