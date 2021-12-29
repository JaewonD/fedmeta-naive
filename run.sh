CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset ICHAR --note 0 --testid $(($1 * 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset ICHAR --note 0 --testid $(($1 * 2 + 1)) &
wait
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset ICHAR --note 0 --testid $(($1 * 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset ICHAR --note 0 --testid $(($1 * 2 + 1)) &
wait
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset ICHAR --note 0 --testid $(($1 * 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset ICHAR --note 0 --testid $(($1 * 2 + 1)) &
wait
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset ICSR --note 0 --testid $(($1 * 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset ICSR --note 0 --testid $(($1 * 2 + 1)) &
wait
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset ICSR --note 0 --testid $(($1 * 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset ICSR --note 0 --testid $(($1 * 2 + 1)) &
wait
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset ICSR --note 0 --testid $(($1 * 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset ICSR --note 0 --testid $(($1 * 2 + 1)) &
wait