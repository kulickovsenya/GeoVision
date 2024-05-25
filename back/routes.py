from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .config import cfg
from .analysis import main as analyse


def set_routes(app):
    @app.post('/analyse')
    async def make_all_operations():
        print('Производим манипуляции с изображением')
        filename = r"GeoVision_dataset\well_1.PDF"
        try:
            analyse(filename)
        except Exception as e:
            raise Exception('Ошибка модуля анализа: ' + str(e))

        return {}
