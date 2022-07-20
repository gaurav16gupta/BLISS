taskset -c 1-64 python3 index1B.py --node=0 --R=4 --gpu=0 --load_epoch=5 --k2=2 
taskset -c 1-64 python3 index1B.py --node=1 --R=4 --gpu=0 --load_epoch=5 --k2=2 
taskset -c 1-64 python3 index1B.py --node=2 --R=4 --gpu=0 --load_epoch=5 --k2=2 
taskset -c 1-64 python3 index1B.py --node=3 --R=4 --gpu=0 --load_epoch=5 --k2=2
taskset -c 1-64 python3 index1B.py --node=4 --R=4 --gpu=0 --load_epoch=5 --k2=2 
taskset -c 1-64 python3 index1B.py --node=5 --R=4 --gpu=0 --load_epoch=5 --k2=2 
taskset -c 1-64 python3 index1B.py --node=6 --R=4 --gpu=0 --load_epoch=5 --k2=2 
taskset -c 1-64 python3 index1B.py --node=7 --R=4 --gpu=0 --load_epoch=5 --k2=2 

# mkdir node_1 && mkdir node_2 && mkdir node_3 && mkdir node_4 && mkdir node_5 && mkdir node_6 && mkdir node_7

