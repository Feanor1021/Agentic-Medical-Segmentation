services:
  orchestrator:
    build:
      context: .
      dockerfile: orchestrator/Dockerfile
    ports:
      - "7860:7860"
    environment:
      - VLM_URL=http://vlm:8001
      - LLM_URL=http://llm:8002
      - TOOL_URL_TOTALSEG=http://tool_totalseg:8011
      - TOOL_URL_VOXTELL=http://tool_voxtell:8012
      - TOOL_URL_BIOMEDPARSE=http://tool_biomedparse:8013
    volumes:
      - ./outputs:/app/outputs
    depends_on:
      - vlm
      - llm
  vlm:
    build:
      context: .
      dockerfile: services/vlm/Dockerfile
    ports:
      - "8001:8001"
    environment:
      - HULU_MODEL_ID=ZJU-AI4H/Hulu-Med-7B
      - HF_HOME=/models/hf
      - TRANSFORMERS_CACHE=/models/hf
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
  llm:
    build:
      context: .
      dockerfile: services/llm/DockerFile
    ports:
      - "8002:8002"
    environment:
      - QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
      - HF_HOME=/models/hf
      - TRANSFORMERS_CACHE=/models/hf
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
  tool_totalseg:
    build:
      context: .
      dockerfile: services/tool_totalseg/Dockerfile
    ports:
      - "8011:8011"
  tool_voxtell:
    build:
      context: .
      dockerfile: services/tool_voxtell/Dockerfile
    ports:
      - "8012:8012"
  tool_biomedparse:
    image: hariskhan687/biomedparse-tool:amd64
    ports:
      - "8013:8013"
    volumes:
      - ./services/tool_biomedparse/api.py:/app/api.py
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
