#!/bin/bash

for d in manila belem nairobi guadalupe mumbai; do
  echo "Evaluating results for $d"
  python3 -m scripts.eval config/$d/env/default.yaml       config/$d/eval_random.yaml models/$d/default       data/results/$d/random.pickle    &> /dev/null &
  python3 -m scripts.eval config/$d/env/noise_unaware.yaml config/$d/eval_random.yaml models/$d/noise_unaware data/results/$d/random_nu.pickle &> /dev/null &
  python3 -m scripts.eval config/$d/env/default.yaml       config/$d/eval_real.yaml   models/$d/default       data/results/$d/real.pickle      &> /dev/null &
  python3 -m scripts.eval config/$d/env/noise_unaware.yaml config/$d/eval_real.yaml   models/$d/noise_unaware data/results/$d/real_nu.pickle   &> /dev/null &
  wait
done

echo "Evaluating enhancements"
python3 -m scripts.eval config/belem/env/no_bridge.yaml            config/belem/eval_random.yaml models/belem/no_bridge            data/results/belem/enhancements/no_bridge.pickle            &> /dev/null &
python3 -m scripts.eval config/belem/env/no_commutation.yaml       config/belem/eval_random.yaml models/belem/no_commutation       data/results/belem/enhancements/no_commutation.pickle       &> /dev/null &
python3 -m scripts.eval config/belem/env/no_enhancements.yaml      config/belem/eval_random.yaml models/belem/no_enhancements      data/results/belem/enhancements/no_enhancements.pickle      &> /dev/null &
python3 -m scripts.eval config/belem/env/no_front_layer_swaps.yaml config/belem/eval_random.yaml models/belem/no_front_layer_swaps data/results/belem/enhancements/no_front_layer_swaps.pickle &> /dev/null &
python3 -m scripts.eval config/belem/env/default.yaml              config/belem/eval_random.yaml models/belem/no_embeddings        data/results/belem/enhancements/no_embeddings.pickle        &> /dev/null &
wait

echo "Evaluating episodes"
python3 -m scripts.eval config/nairobi/env/default.yaml config/nairobi/eval_random.yaml models/nairobi/default data/results/nairobi/episodes/deterministic.pickle --deterministic &> /dev/null
for ep in "1" "2" "4" "8" "16"; do
  echo "$ep episodes"
  python3 -m scripts.eval config/nairobi/env/default.yaml config/nairobi/eval_random.yaml models/nairobi/default data/results/nairobi/episodes/stochastic_${ep}ep.pickle -e $ep &> /dev/null
done
