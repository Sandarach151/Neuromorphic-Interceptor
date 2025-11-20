1. ssh -Y -C raid@192.168.55.1
2. scp raid@192.168.55.1:/home/raid/Documents/analysis.ipynb .
3. metavision_platform_info
4. metavision_viewer
5. cmake --build . -j"$(nproc)"
6. Check prophesee docs to do noise cancellation automatically if needed in 
7. wifi connection - sudo nmtui
future