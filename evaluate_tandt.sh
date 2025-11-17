# tandt dataset evaluation script

# train
# render LocoGS
python render.py -s data/tandt/train -m output/train_hybrid
# compute error metrics on renderings
python metrics.py -m output/train_hybrid

# truck
# render LocoGS
python render.py -s data/tandt/truck -m output/truck_hybrid
# compute error metrics on renderings
python metrics.py -m output/truck_hybrid
