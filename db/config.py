from sqlalchemy import create_engine
from environs import Env

env = Env()
env.read_env()

# Database
DB_NAME = 'leq_db'
DB_HOST = 'localhost:5432'
DB_PASSWORD = ''
DB_USER = 'postgres'
DB_ENGINE = 'postgresql+psycopg2'

DB_ENGINE = create_engine('{0}://{1}:{2}@{3}/{4}'.format(DB_ENGINE, DB_USER,
                                                      DB_PASSWORD, DB_HOST, DB_NAME), pool_pre_ping=True)
