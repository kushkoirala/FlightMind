Set-Location "D:\FlightMind"
python -u train/finetune.py `
    --checkpoint "checkpoints/v2_cloud/best.pt" `
    --lora-rank 16 `
    --lora-alpha 32 `
    --max-steps 2000 `
    --batch-size 2 `
    --grad-accum 2 `
    --lr 2e-4 `
    --min-lr 2e-5 `
    --warmup-steps 100 `
    --seq-len 512 `
    --log-every 10 `
    --checkpoint-every 500 `
    --device cuda
