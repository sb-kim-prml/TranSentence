# TranSentence: Speech-to-speech Translation via Language-agnostic Sentence-level Speech Encoding without Language-parallel Data

## Requirements and Installation
- PyTorch version >= 1.10.0
- Python version >= 3.8
- Based on [fairseq](https://github.com/facebookresearch/fairseq/tree/main)
- Install fairseq following the installation guide of fairseq

## Preprocessing


## Training
```
MODEL_DIR=./logs/transentence
CUDA_VISIBLE_DEVICES=0,1 fairseq-train ${DATA_ROOT} \
  --config-yaml ./configs/config.yaml \
  --task sem_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch s2ut_transformer_dec --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset valid \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 200000 --max-tokens 20000 --max-target-positions 3000 --update-freq 8 \
  --seed 1 --fp16 --num-workers 8 \
  --validate-interval 5 --save-interval 20 \
  --tensorboard-logdir ${MODEL_DIR} \
  --input_split True
```

## Inference
Speech-to-unit translation
```
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_ROOT \
  --config-yaml ./configs/config.yaml \
  --task sem_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --path $MODEL_DIR/checkpoint_best.pt  --gen-subset test \
  --max-tokens 50000 \
  --beam 10 --max-len-a 1 \
  --results-path $MODEL_DIR/results \
  --arch s2ut_transformer_dec \
  --input_split True \
```


Unit-to-waveform conversion
```
CUDA_VISIBLE_DEVICES=0 python examples/speech_to_speech/generate_waveform_from_code_eval.py \
--in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt \
--vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
--results-path ${RESULTS_PATH}/wavs --dur-prediction
```