# import os
# from contextlib import asynccontextmanager
# from logging import getLogger

# from dotenv import load_dotenv  # Load .env variables
# from fastapi import FastAPI, Response, status

# from bind import Bind, BindInput
# from meta import Meta

# # Load environment variables
# # load_dotenv()

# bind: Bind
# meta_config: Meta
# logger = getLogger('uvicorn')


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global bind
#     global meta_config

#     logger.info("Starting lifespan context")
#     cuda_env = os.getenv("ENABLE_CUDA", "false").lower()
#     cuda_support = False
#     cuda_core = "cpu"

#     if cuda_env in ["true", "1"]:
#         cuda_support = True
#         cuda_core = os.getenv("CUDA_CORE", "cuda:0")
#         logger.info(f"CUDA_CORE set to {cuda_core}")
#     else:
#         logger.info("Running on CPU")

#     try:
#         bind = Bind(cuda_support, cuda_core)
#         meta_config = Meta()
#         logger.info("Model initialization complete")
#     except Exception as e:
#         logger.exception("Failed to initialize Bind or Meta")
#         raise e  # Ensure the error propagates, and FastAPI logs it

#     yield

# app = FastAPI(lifespan=lifespan)


# @app.get("/.well-known/live", response_class=Response)
# @app.get("/.well-known/ready", response_class=Response)
# @app.head("/.well-known/live", response_class=Response)
# @app.head("/.well-known/ready", response_class=Response)
# async def live_and_ready(response: Response):
#     response.status_code = status.HTTP_204_NO_CONTENT


# @app.get("/meta")
# async def meta():
#     return await meta_config.get()


# @app.post("/vectorize")
# async def vectorize(payload: BindInput, response: Response):
#     try:
#         result = await bind.vectorize(payload)
#         return {
#             "textVectors": result.text_vectors,
#             "imageVectors": result.image_vectors,
#             "audioVectors": result.audio_vectors,
#             "videoVectors": result.video_vectors,
#             "imuVectors": result.imu_vectors,
#             "depthVectors": result.depth_vectors,
#             "thermalVectors": result.thermal_vectors,
#         }
#     except Exception as e:
#         logger.exception('Something went wrong while vectorizing data.')
#         response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
#         return {"error": str(e)}


import os
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI, Response, status

from bind import Bind, BindInput
from meta import Meta

bind: Bind
meta_config: Meta
logger = getLogger('uvicorn')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bind
    global meta_config

    logger.info("Starting lifespan context")
    # Set CUDA configuration directly without environment variables
    cuda_support = True
    # Change "cuda:0" if you want to specify a different GPU, e.g., "cuda:1"
    cuda_core = "cuda:0"
    logger.info(f"CUDA_CORE set to {cuda_core}")

    try:
        bind = Bind(cuda_support, cuda_core)
        meta_config = Meta()
        logger.info("Model initialization complete")
    except Exception as e:
        logger.exception("Failed to initialize Bind or Meta")
        raise e  # Ensure the error propagates, and FastAPI logs it

    yield

app = FastAPI(lifespan=lifespan)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
@app.head("/.well-known/live", response_class=Response)
@app.head("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
async def meta():
    return await meta_config.get()


@app.post("/vectorize")
async def vectorize(payload: BindInput, response: Response):
    try:
        result = await bind.vectorize(payload)
        return {
            "textVectors": result.text_vectors,
            "imageVectors": result.image_vectors,
            "audioVectors": result.audio_vectors,
            "videoVectors": result.video_vectors,
            "imuVectors": result.imu_vectors,
            "depthVectors": result.depth_vectors,
            "thermalVectors": result.thermal_vectors,
        }
    except Exception as e:
        logger.exception('Something went wrong while vectorizing data.')
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
