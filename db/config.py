from sqlalchemy import create_engine
from environs import Env

env = Env()
env.read_env()

DB_NAME = env.str('DB_NAME')
DB_HOST = env.str('DB_HOST')
DB_PASSWORD = env.str('DB_PASSWORD')
DB_USER = env.str('DB_USER')
DB_ENGINE = env.str('DB_ENGINE')

ENGINE = create_engine('{0}://{1}:{2}@{3}/{4}'.format(DB_ENGINE, DB_USER,
                                                      DB_PASSWORD, DB_HOST, DB_NAME), pool_pre_ping=True)

ENGINE.connect()
