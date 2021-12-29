CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2))_d0 &
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2))_d1 &
wait
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2))_d2 &
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2))_d3 &
wait
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2))_d4 &
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2 + 1))_d0 &
wait
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2 + 1))_d1 &
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2 + 1))_d2 &
wait
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2 + 1))_d3 &
CUDA_VISIBLE_DEVICES=$(($1+5)) python main.py --algorithm default --dataset HHAR --note 0 --testid u$(($1 * 2 + 1))_d4 &
wait