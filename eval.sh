for method in baseline sink pitome
do 

   python benchmark/perplexity.py --experiment  $method --overwrite --window_size 1024 
done
