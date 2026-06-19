$env:PYTHONUNBUFFERED = "1"
Set-Location D:\FlightMind
python -u train/pretrain.py --depth 8 --device cuda --batch-size 4 --tokens-per-step 262144 --max-steps 5000 --warmup-steps 200 --lr 6e-4 --min-lr 6e-5 --eval-every 250 --checkpoint-every 250 --log-every 10
