from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Import models here to register with SQLAlchemy
from .user import User
