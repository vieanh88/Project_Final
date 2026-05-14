# [DATA PIPELINE]
prepare_vivoice/step1 → step2 → step3 → step4 → step5 → step6
prepare_ngan/step1 → step2
prepare_ood/step1
plbert_v2/step1 → step2 → verify_plbert_checkpoint.py

# [TRAINING]
train_wrapper.py --stage 1  →  --stage 2  →  --stage 3

# [INFERENCE]
nlp_generator.py  →  (tắt vLLM)  →  create_mean_style.py  →  tts_generator.py
                                                                    ↓
                                                         output_ghost_story.wav