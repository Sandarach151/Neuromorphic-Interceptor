1. ssh -Y -C raid@192.168.55.1
2. scp raid@192.168.55.1:/home/raid/Documents/analysis.ipynb .
3. metavision_platform_info
4. metavision_viewer
5. cmake --build . -j"$(nproc)"
6. Check prophesee docs to do noise cancellation automatically if needed in future
7. sudo nmtui
8. bullet details
    - Magnus (17.70 +- 0.31) m/s
    - Accustrike (15.69 +- 0.34) m/s
    - Hole Bullet 15.74 m/s
    - Black Bullet 20.90 m/s
9. sudo dmesg -w