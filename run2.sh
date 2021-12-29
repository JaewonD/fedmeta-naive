CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset WESAD --note 0 --testid $(($1 * 3)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset WESAD --note 0 --testid $(($1 * 3 + 1)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm default --dataset WESAD --note 0 --testid $(($1 * 3 + 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset WESAD --note 0 --testid $(($1 * 3)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset WESAD --note 0 --testid $(($1 * 3 + 1)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm meta --dataset WESAD --note 0 --testid $(($1 * 3 + 2)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset WESAD --note 0 --testid $(($1 * 3)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset WESAD --note 0 --testid $(($1 * 3 + 1)) &
CUDA_VISIBLE_DEVICES=$1 python main.py --algorithm mix --dataset WESAD --note 0 --testid $(($1 * 3 + 2)) &
wait
