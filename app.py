from flask import Flask

from api.agent import agent_bp
from api.kb import kb_bp
from api.system import system_bp
from api.tools import tools_bp
from api.upload import upload_bp
from config import MAX_FILE_SIZE
from utils.kb_db import ensure_kb_db


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

    app.register_blueprint(upload_bp)
    app.register_blueprint(kb_bp)
    app.register_blueprint(system_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(agent_bp)

    ensure_kb_db()
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
