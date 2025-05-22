You would need to install a bunch of things. And currently only supports Mac

python fastrtc_server.py \            
  --model small \
  --lan en \
  --min-chunk-size 1.0 \
  --host 0.0.0.0 \
  --port 9880 \
 --backend mlx-whisper
