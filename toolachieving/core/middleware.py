# -*- coding: utf-8 -*-
import time, uuid
from flask import request, jsonify, g

API_KEY = None

def install_middlewares(app):
    @app.before_request
    def _before():
        g.req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        g._t0 = time.time()
        # if API_KEY and not request.path.startswith("/v1/health"):
        #     if request.headers.get("X-API-Key") != API_KEY:
        #         return jsonify({"error":"Unauthorized"}), 401

    @app.after_request
    def _after(resp):
        if hasattr(g, "_t0"):
            resp.headers["X-Request-Id"] = g.req_id
            resp.headers["X-Elapsed-ms"] = str(int((time.time()-g._t0)*1000))
        return resp
