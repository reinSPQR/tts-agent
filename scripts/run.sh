docker run -p 8000:8000 \
    -v ./checkpoints:/app/checkpoints \
    -e MODEL_REPO=bosonai/higgs-audio-v2-generation-3B-base \
    -e AUDIO_TOKENIZER_REPO=bosonai/higgs-audio-v2-tokenizer \
    tts-agent:latest
