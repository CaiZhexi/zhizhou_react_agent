# -*- coding: utf-8 -*-
from flask import jsonify

def install_error_handlers(app):
    @app.errorhandler(Exception)
    def _fallback(e):
        return jsonify({"error": "InternalError", "detail": str(e)}), 500
