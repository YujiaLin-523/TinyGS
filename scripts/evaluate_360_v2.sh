# 360_v2 dataset evaluation script

# bicycle
# render LocoGS
python render.py -s data/360_v2/bicycle -m output/bicycle_hybrid
# compute error metrics on renderings
python metrics.py -m output/bicycle_hybrid

# bonsai
# render LocoGS
python render.py -s data/360_v2/bonsai -m output/bonsai_hybrid
# compute error metrics on renderings
python metrics.py -m output/bonsai_hybrid

# counter
# render LocoGS
python render.py -s data/360_v2/counter -m output/counter_hybrid
# compute error metrics on renderings
python metrics.py -m output/counter_hybrid

# garden
# render LocoGS
python render.py -s data/360_v2/garden -m output/garden_hybrid
# compute error metrics on renderings
python metrics.py -m output/garden_hybrid

# kitchen
# render LocoGS
python render.py -s data/360_v2/kitchen -m output/kitchen_hybrid
# compute error metrics on renderings
python metrics.py -m output/kitchen_hybrid

# room
# render LocoGS
python render.py -s data/360_v2/room -m output/room_hybrid
# compute error metrics on renderings
python metrics.py -m output/room_hybrid

# stump
# render LocoGS
python render.py -s data/360_v2/stump -m output/stump_hybrid
# compute error metrics on renderings
python metrics.py -m output/stump_hybrid