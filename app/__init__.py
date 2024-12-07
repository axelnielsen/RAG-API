from flask import Flask
from flasgger import Swagger
from app.config import Config
from app.log_config import setup_logging
import logging

setup_logging()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    swagger = Swagger(app, template_file="./swagger/generate.yaml")
    
    with app.app_context():
        from app.routes import main_bp
        app.register_blueprint(main_bp)
    
    return app

logger = logging.getLogger(__name__)
logger.info('Flask application initialized')
