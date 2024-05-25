from fastapi import FastAPI
from .routes import set_routes
from .config import cfg

app = FastAPI(docs_url=cfg.URL_SWAGGER)

set_routes(app)