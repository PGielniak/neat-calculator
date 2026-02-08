#! /bin/bash

docker build -t neat-calculator-pilot:latest .

docker tag neat-calculator-pilot:latest patryklinuksiarz/neat-calculator-ml:latest

docker push patryklinuksiarz/neat-calculator-ml:latest