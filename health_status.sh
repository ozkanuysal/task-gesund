#!/bin/bash

container_names=("peaceful_antonelli" "zen_brattain")

while true; do
    for name in "${container_names[@]}"; do
        container_id=$(docker ps -qf "name=$name")
        
        image_name=$(docker inspect -f '{{.Config.Image}}' $container_id)
        echo "Image Name: $image_name"
        
        echo "-----------------------"

        utilization=$(docker stats $container_id --no-stream --format "CPU: {{.CPUPerc}}, Memory: {{.MemUsage}}")
        echo "Utilization of $name: $utilization"

        echo "-----------------------"

        status=$(docker inspect -f '{{.State.Status}}' $container_id)
        echo "Status of $name: $status"

        echo "-----------------------"

        start_time=$(docker inspect -f '{{.State.StartedAt}}' $container_id)
        echo "Start Time: $start_time"

        echo "-----------------------"


    done
    sleep 5
done