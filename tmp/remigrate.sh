#!/bin/bash
cd /home/johnfaqpc/programming/AI_Pedsim/Geo_scenario/Topo_HouseGAN
for p in plan_*; do
    if [ -d "$p" ]; then
        echo "Processing $p..."
        for cat in geo dataswarm heatmap_density heatmap_speed spawn_exit trajectory_line; do
            if [ -d "$p/$cat" ]; then
                mkdir -p "$cat/$p"
                mv "$p/$cat/"* "$cat/$p/"
                rmdir "$p/$cat"
            fi
        done
        rmdir "$p"
    fi
done
echo "Done."
