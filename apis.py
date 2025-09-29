import io
import os
import time
import uuid
import json
import asyncio
import base64
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
import uvicorn
from typing import AsyncGenerator, Dict, Any, Optional, List
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

from boson_multimodal.serve.serve_engine import HiggsAudioStreamerDelta, HiggsAudioResponse, HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from oai_models import ChatCompletionMessageParam, ChatCompletionRequest
from schema import AudioChunk, AudioGenerationRequest, ResponseStatus, TaskStatus


async def send_chunk(chunk_obj):
    """Helper to send chunk with optimized JSON serialization"""
    # Use model_dump_json() to properly handle datetime serialization
    chunk_json = chunk_obj.model_dump_json()
    yield f"data: {chunk_json}\n\n"


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
        model_dir = os.path.join("./checkpoints", config.model_repo)
        logger.info(f"Downloading {config.model_repo} to {model_dir}")
        snapshot_download(repo_id=config.model_repo, local_dir=model_dir)

        audio_tokenizer_dir = os.path.join("./checkpoints", config.audio_tokenizer_repo)
        logger.info(f"Downloading {config.audio_tokenizer_repo} to {audio_tokenizer_dir}")
        snapshot_download(repo_id=config.audio_tokenizer_repo, local_dir=audio_tokenizer_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        serve_engine = HiggsAudioServeEngine(model_dir, audio_tokenizer_dir, device=device)
        
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
async def chat_completions(request: AudioGenerationRequest, original_request: Request):
    async def optimized_stream_generator():
        try:
            task_id = str(uuid.uuid4())

            streamer: AsyncGenerator[HiggsAudioStreamerDelta, None] = serve_engine.generate_delta_stream(
                chat_ml_sample=ChatMLSample(messages=request.messages),
                max_new_tokens=request.max_completion_tokens or 1024,
                temperature=request.temperature or 0.3,
                top_p=request.top_p or 0.95,
                top_k=request.top_k or 50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                seed=request.seed or 42,
            )
            
            audio_token_buffer = []
            chunk_size = 64
            
            async for delta in streamer:
                if delta.audio_tokens is not None:
                    audio_token_buffer.append(delta.audio_tokens)

                is_final_chunk = delta.text == "<|eot_id|>"
                
                if len(audio_token_buffer) >= chunk_size or is_final_chunk:
                    audio_chunk = torch.stack(audio_token_buffer[:chunk_size], dim=1)
                    num_codebooks = audio_chunk.shape[0]
                    
                    vq_code = revert_delay_pattern(audio_chunk).clip(0, serve_engine.audio_codebook_size - 1)
                    wv_numpy = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                    print(wv_numpy.tolist())
                
                    # Calculate how many tokens to keep for next chunk
                    # We need to preserve the tokens that were cut off by the delay pattern
                    # Keep the last (num_codebooks - 1) tokens to maintain continuity
                    tokens_to_keep = num_codebooks - 1
                    audio_token_buffer = audio_token_buffer[chunk_size - tokens_to_keep:]

                    if wv_numpy.shape[0] > 0:
                        chunk_obj = AudioChunk(
                            id=task_id,
                            content=delta.text,
                            audio=wv_numpy.tolist(),
                            status=ResponseStatus(status=TaskStatus.COMPLETED, progress=99.0, estimated_wait_time=None),
                            finish_reason=None,
                            sampling_rate=serve_engine.audio_tokenizer.sampling_rate,
                        )

                    if is_final_chunk:
                        chunk_obj = AudioChunk(
                            id=task_id,
                            content=delta.text,
                            audio=[],
                            status=ResponseStatus(status=TaskStatus.COMPLETED, progress=99.0, estimated_wait_time=None),
                            finish_reason="stop",
                            sampling_rate=serve_engine.audio_tokenizer.sampling_rate,
                        )

                    async for chunk in send_chunk(chunk_obj):
                        yield chunk

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in image stream generator for task {task_id}: {e}", exc_info=True)
            try:
                async for chunk in send_chunk(AudioChunk(
                    id=task_id,
                    content=f"Stream error: {str(e)}",
                    status=ResponseStatus(status=TaskStatus.FAILED, progress=0.0),
                    finish_reason="error"
                )):
                    yield chunk
                yield "data: [DONE]\n\n"
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup for task {task_id}: {cleanup_error}", exc_info=True)
    
    return StreamingResponse(optimized_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
