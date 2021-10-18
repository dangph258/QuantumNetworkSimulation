#!/bin/bash
DEPO=(100 1000 10000)
DEPH=(100 1000 10000)
DISTANCE=(1 5 10 15 20 25 30)
for k in ${DEPO[@]}
do
    for t in ${DEPH[@]}
    do
        file=depo$((k))_deph$((t))_v1.txt
        > $file 
        echo $file
        for i in ${DISTANCE[@]}
        do
            #distance=$((i))
            # echo $k $t $i
            #filename=ev1_$i_$k_$t.txt
            #echo $filename
            python3 ev1_teleportation.py $i $k $t $file
        done
    done
done