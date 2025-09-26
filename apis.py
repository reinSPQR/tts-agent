import io
import os
import time
import uuid
import json
import asyncio
import base64
import uvicorn
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import heapq
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request
import torch
import torchaudio
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download

from boson_multimodal.serve.serve_engine import HiggsAudioResponse, HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from oai_models import ChatCompletionRequest


load_dotenv()

# Configuration
@dataclass
class Config:
    """Application configuration"""
    hf_token: str = os.getenv("HF_TOKEN", "")
    model_repo: str = os.getenv("MODEL_REPO", "bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer_repo: str = os.getenv("AUDIO_TOKENIZER_REPO", "bosonai/higgs-audio-v2-tokenizer")
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    sync_timeout: int = int(os.getenv("SYNC_TIMEOUT", "600"))
    cleanup_interval: int = int(os.getenv("CLEANUP_INTERVAL", "300"))
    task_ttl: int = int(os.getenv("TASK_TTL", "3600"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    default_processing_time: int = int(os.getenv("DEFAULT_PROCESSING_TIME", "25"))

config = Config()

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper()))
logger = logging.getLogger(__name__)

serve_engine = None
# queue_manager: Optional[ProductionQueueManager] = None

async def initialize_serve_engine():
    """Initialize the FLUX.1-Krea-dev image generation pipeline"""
    global serve_engine
    try:
        logger.info(f"Downloading {config.model_repo}...")
        snapshot_download(repo_id=config.model_repo)
        logger.info(f"Downloading {config.audio_tokenizer_repo}...")
        snapshot_download(repo_id=config.audio_tokenizer_repo)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        serve_engine = HiggsAudioServeEngine(config.model_repo, config.audio_tokenizer_repo, device=device)
        
        logger.info("Serve engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize serve engine: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # global queue_manager
    
    # Startup
    logger.info("Starting up the text-to-speech API...")
    
    # Initialize pipeline
    await initialize_serve_engine()
    
    # Initialize queue manager
    # queue_manager = ProductionQueueManager(
    #     max_queue_size=config.max_queue_size, 
    #     max_concurrent=config.max_concurrent
    # )
    # await queue_manager.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down the image generation API...")
    # if queue_manager:
    #     await queue_manager.stop()


app = FastAPI(
    title="Audio text-to-speech API",
    description="A FastAPI-based text-to-speech API",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, original_request: Request):
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=request.messages),
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)