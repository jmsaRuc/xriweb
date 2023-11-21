import os
from importlib import metadata
from typing import Optional

from fastapi import FastAPI, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, UJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from xriweb.settings import settings
from xriweb.web.api.router import api_router
from xriweb.web.lifetime import register_shutdown_event, register_startup_event
from xriweb.web.reshaperun import reshapeAndPlot


def get_app() -> FastAPI:
    """

    Get FastAPI application.
    This is the main constructor of an application.
    :return: application.
    """
    app = FastAPI(
        title="xriweb",
        version=metadata.version("xriweb"),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static/", StaticFiles(directory=settings.static_dir), name="static")
    temp_fileU_List = []
    temp_fileu_listname = []
    templates = Jinja2Templates(directory=settings.template_dir)

    @app.get("/index/", response_class=HTMLResponse)
    async def index(request: Request, hx_request: Optional[str] = Header(None)):
        context = {"request": request}
        if hx_request:
            context = {"request": request}
            for filename in os.listdir(settings.modelres_dir):
                os.remove(settings.modelres_dir / filename)
            for idx, temp_file_u in enumerate(temp_fileU_List):
                image_p = Image.open(temp_file_u, "r")
                image = reshapeAndPlot(image_p)
                save_as = settings.modelres_dir / temp_fileu_listname[idx]
                image.save(save_as)
                image_p.close()
                image.close()
            temp_fileU_List.clear()
            temp_fileu_listname.clear()
            imgs = os.listdir(settings.modelres_dir)
            context = {"request": request, "imgs": imgs}
            return templates.TemplateResponse("results.html", context)
        else:
            for filename in os.listdir(settings.upload_dir):
                os.remove(settings.upload_dir / filename)
        return templates.TemplateResponse("index.html", context)

    @app.post("/uploadfile/")
    async def create_upload_file(file_uploads: list[UploadFile]):
        for file_upload in file_uploads:
            data = await file_upload.read()
            save_to = settings.upload_dir / file_upload.filename
            with open(save_to, "wb") as f:
                f.write(data)
        temp_fileU_List.append(save_to)
        temp_fileu_listname.append(file_upload.filename)
        return {"filenames": [file_upload.filename for f in file_uploads]}

    # Adds startup and shutdown events.
    register_startup_event(app)
    register_shutdown_event(app)

    # Main router for the API.
    app.include_router(router=api_router, prefix="/api")

    return app
