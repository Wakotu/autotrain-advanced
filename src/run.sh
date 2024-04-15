#!/usr/bin/env bash

PROJECT_NAME='local-test'
MODEL_NAME='mistralai/Mistral-7B-v0.1'
LEARNING_RATE=2e-4
NUM_EPOCHS=4
BATCH_SIZE=1
BLOCK_SIZE=1024
TRAINER=sft
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION=4
USE_FP16=True
USE_PEFT=True
USE_INT4=True
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.045

~/.conda/envs/autotrain/bin/python main.py llm \
	--train \
	--trainer $TRAINER \
	--model ${MODEL_NAME} \
	--project-nam ${PROJECT_NAME} \
	--data-path "timdettmers/openassistant-guanaco" \
	--text-column text \
	--lr ${LEARNING_RATE} \
	--batch-size ${BATCH_SIZE} \
	--epochs ${NUM_EPOCHS} \
	--block-size ${BLOCK_SIZE} \
	--warmup-ratio ${WARMUP_RATIO} \
	--lora-r ${LORA_R} \
	--lora-alpha ${LORA_ALPHA} \
	--lora-dropout ${LORA_DROPOUT} \
	--weight-decay ${WEIGHT_DECAY} \
	--gradient-accumulation ${GRADIENT_ACCUMULATION} \
	--mp fp16 \
	$([[ "$USE_PEFT" == "True" ]] && echo "--use-peft") \
	--quantization int4 \
	--backend local-cli \
	--username Wakotu
