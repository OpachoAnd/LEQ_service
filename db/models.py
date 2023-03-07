from sqlalchemy import (
    Column,
    Text,
    BigInteger,
    ForeignKey
)

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    user_id = Column('user_id', BigInteger, primary_key=True, autoincrement=True)
    source_id = Column('source_id', BigInteger, ForeignKey('source.source_id'), autoincrement=True)
    e_mail = Column('e_mail', Text)


class Source(Base):
    __tablename__ = 'source'
    source_id = Column('source_id', BigInteger, primary_key=True, autoincrement=True)
    video_id = Column('video_id', BigInteger, ForeignKey('video.video_id'), autoincrement=True)
    path_original_video = Column('path_original_video', BigInteger)
    path_audio = Column('path_audio', BigInteger)


class Video(Base):
    __tablename__ = 'video'
    video_id = Column('video_id', BigInteger, primary_key=True, autoincrement=True)
    path_video_NeRF = Column('path_video_NeRF', Text)
    path_video_W2L_e = Column('path_video_W2L_e', Text)
