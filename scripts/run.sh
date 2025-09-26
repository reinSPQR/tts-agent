docker run -p 8000:8000 \
    -v ${PWD}/checkpoints:/app/checkpoints \
    -e MODEL_REPO=checkpoints/higgs-audio-v2-generation-3B-base \
    -e AUDIO_TOKENIZER_REPO=checkpoints/higgs-audio-v2-tokenizer \
    tts-agent:latest
