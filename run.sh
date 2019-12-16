python main2.py --mode classification --method none --loss-function bce --exp baseline --epoch 50

python main2.py --mode classification --method none --loss-function bce --exp baseline_smoothing --epoch 100 --tricks cut-out

python main2.py --mode classification --method none --loss-function bce --exp baseline_cut-out --epoch 100 --tricks smoothing

python main2.py --mode classification --method none --loss-function bce --exp baseline_all --epoch 100 --tricks all


#python main2.py --mode segmentation --method a1d2v3 --loss-function cross_entropy --exp seg_base --epoch 200

#python main2.py --mode segmentation --method adv --loss-function cross_entropy --exp seg_adv --epoch 100

#python main2.py --mode segmentation --method adv --loss-function cross_entropy 
